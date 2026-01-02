"""
Traffic Data Processing Pipeline

This module provides a complete pipeline for processing traffic monitoring data:
- Categorizes vehicle IDs
- Finds and merges fragmented vehicle trajectories
- Assigns unified IDs in chronological order
- Consolidates merged records
"""

import argparse
import pandas as pd
from utils.transformer import categorize_ids, find_merging_pairs, build_merge_chains, apply_merges_to_summary
from utils.loader import load_data_from_database

def process_all_sessions(df, category_filter=None, time_gap_limit=1.5, space_gap_limit=40, size_sim_limit=0.2, verbose=True):
    """
    Process all sessions in the dataframe: categorize, find merges, and apply unified IDs.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw traffic data with session_id column
    category_filter : list, optional
        Categories to consider for merging (default: ['Noise', 'Partial', 'Static'])
    time_gap_limit : float
        Maximum time gap in seconds for merging
    space_gap_limit : float
        Maximum spatial distance for merging
    size_sim_limit : float
        Maximum size difference ratio for merging
    verbose : bool
        If True, print detailed progress for each session (default: True)
        
    Returns:
    --------
    pd.DataFrame
        Complete final summary with unified_id and Merged categories for all sessions
    """
    if category_filter is None:
        category_filter = ['Partial', 'Static', 'Noise', 'Ghost']
    
    # Get all unique sessions
    session_ids = df['session_id'].unique()
    
    all_summaries = []
    unified_id_counter = 1
    
    if verbose:
        print(f"Processing {len(session_ids)} sessions...")
        print("=" * 80)
    
    for session_idx, session_id in enumerate(session_ids, 1):
        if verbose:
            print(f"\n📍 Session {session_idx}/{len(session_ids)} (session_id={session_id})")
            print("-" * 80)
        
        # Filter data for this session
        session_df = df[df['session_id'] == session_id].copy()
        
        # Step 1: Categorize IDs for this session
        if verbose:
            print("  Step 1: Categorizing IDs...")
        session_summary = categorize_ids(session_df)
        
        if verbose:
            category_counts = session_summary.groupby('category').size()
            print("  Category distribution:")
            for cat, count in category_counts.items():
                print(f"    • {cat:15s}: {count:4d} IDs")
        
        # Step 2: Find merging pairs
        if verbose:
            print("\n  Step 2: Finding merge pairs...")
        merge_results = find_merging_pairs(
            session_summary, 
            category_filter=category_filter,
            time_gap_limit=time_gap_limit,
            space_gap_limit=space_gap_limit,
            size_sim_limit=size_sim_limit
        )
        
        if verbose:
            print(f"  Found pairs for merging: {len(merge_results)}")
        
        # Step 3: Build chains and apply merges
        if len(merge_results) > 0:
            if verbose:
                print("\n  Step 3: Building merge chains and consolidating...")
            chains = build_merge_chains(merge_results)
            
            if verbose:
                print(f"  Built {len(chains)} chains")
                if chains:
                    chain_lengths = [len(c) for c in chains]
                    print(f"    • Longest chain: {max(chain_lengths)} IDs")
                    print(f"    • Average chain length: {sum(chain_lengths)/len(chain_lengths):.1f} IDs")
            
            # Apply merges to summary
            session_summary = apply_merges_to_summary(
                session_summary, 
                chains, 
                unified_id_start=unified_id_counter
            )
            
            if verbose:
                merged_count = len(session_summary[session_summary['category'] == 'Merged'])
                unified_count = session_summary['unified_id'].notna().sum()
                print(f"  ✓ Consolidated into {merged_count} merged records")
                print(f"  ✓ Total records with unified_id: {unified_count}")
            
            # Update counter for next session
            unified_id_counter += session_summary['unified_id'].notna().sum() + 1
        else:
            if verbose:
                print("\n  Step 3: No merges found - skipping consolidation")
            # No merges found, just add empty unified_id column
            session_summary['unified_id'] = pd.NA
        
        all_summaries.append(session_summary)
    
    # Combine all sessions
    if verbose:
        print("\n" + "=" * 80)
        print("📊 FINAL SUMMARY")
        print("=" * 80)
    
    final_summary = pd.concat(all_summaries, ignore_index=True)
    
    if verbose:
        print(f"\nTotal records: {len(final_summary):,}")
        print(f"Records with unified_id: {final_summary['unified_id'].notna().sum():,}")
        print(f"Unique unified_ids: {final_summary['unified_id'].nunique()}")
        
        print("\nFinal category distribution:")
        final_category_counts = final_summary['category'].value_counts()
        for cat, count in final_category_counts.items():
            percentage = (count / len(final_summary)) * 100
            print(f"  • {cat:15s}: {count:6d} ({percentage:5.1f}%)")
        
        merged_total = len(final_summary[final_summary['category'] == 'Merged'])
        if merged_total > 0:
            print(f"\n✨ Successfully consolidated {merged_total} merged records across all sessions")
    
    return final_summary


def main():
    """Main execution function for the traffic data processing pipeline."""
    parser = argparse.ArgumentParser(
        description='Process traffic monitoring data: categorize, merge, and assign unified IDs',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Data source arguments
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='data/raw_traffic_data.csv',
        help='Path to input CSV file (default: data/raw_traffic_data.csv)'
    )
    
    parser.add_argument(
        '--from-database',
        action='store_true',
        help='Load data from database instead of CSV file'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/processed_traffic_data.csv',
        help='Path to output CSV file (default: data/processed_traffic_data.csv)'
    )
    
    # Processing parameters
    parser.add_argument(
        '--categories',
        nargs='+',
        default=['Noise', 'Partial', 'Static'],
        help='Categories to consider for merging (default: Noise Partial Static)'
    )
    
    parser.add_argument(
        '--time-gap',
        type=float,
        default=1.5,
        help='Maximum time gap in seconds for merging (default: 1.5)'
    )
    
    parser.add_argument(
        '--space-gap',
        type=float,
        default=40,
        help='Maximum spatial distance in pixels for merging (default: 40)'
    )
    
    parser.add_argument(
        '--size-sim',
        type=float,
        default=0.2,
        help='Maximum size difference ratio for merging (default: 0.2)'
    )
    
    args = parser.parse_args()
    
    # Load data
    print("=" * 60)
    print("TRAFFIC DATA PROCESSING PIPELINE")
    print("=" * 60)
    
    if args.from_database:
        print("\n📊 Loading data from database...")
        df = load_data_from_database()
    else:
        # Interactive prompt if not specified via command line
        print("\n❓ Data source selection:")
        print(f"   Default: Load from {args.input}")
        print(f"   Alternative: Load from database")
        response = input("\nDo you want to load data from the database? (y/N): ").strip().lower()
        
        if response == 'y':
            print("\n📊 Loading data from database...")
            df = load_data_from_database()
        else:
            print(f"\n📁 Loading data from {args.input}...")
            df = pd.read_csv(args.input, parse_dates=['date_time'])
    
    df = df.sort_values(['date_time', 'frame_id'])
    print(f"✓ Loaded {len(df):,} records")
    
    # Create session_id
    print("\n🔢 Creating session IDs...")
    df['session_id'] = (df['frame_id'].diff() < 0).cumsum()
    print(f"✓ Found {df['session_id'].nunique()} sessions")
    
    # Process all sessions
    print("\n🚀 Processing sessions...")
    print(f"   Categories for merging: {', '.join(args.categories)}")
    print(f"   Time gap limit: {args.time_gap}s")
    print(f"   Space gap limit: {args.space_gap} pixels")
    print(f"   Size similarity limit: {args.size_sim}")
    print()
    
    processed_summary = process_all_sessions(
        df,
        category_filter=args.categories,
        time_gap_limit=args.time_gap,
        space_gap_limit=args.space_gap,
        size_sim_limit=args.size_sim
    )
    
    # Save results
    print(f"\n💾 Saving results to {args.output}...")
    processed_summary.to_csv(args.output, index=False)
    print(f"✓ Saved {len(processed_summary):,} records")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"\nTotal records: {len(processed_summary):,}")
    print(f"Records with unified_id: {processed_summary['unified_id'].notna().sum():,}")
    print(f"Unique unified_ids: {processed_summary['unified_id'].nunique()}")
    
    merged_count = len(processed_summary[processed_summary['category'] == 'Merged'])
    if merged_count > 0:
        print(f"\n✨ Successfully consolidated {merged_count} merged records")
    
    print("\n✅ Pipeline completed successfully!")


if __name__ == "__main__":
    main()
