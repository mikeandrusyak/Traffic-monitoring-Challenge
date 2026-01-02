import pg8000
from dotenv import load_dotenv
from google.cloud.sql.connector import Connector, IPTypes
import os
import pandas as pd

def get_conn(connector, instance_connection_name, db_user, db_pass, db_name, ip_type) -> pg8000.dbapi.Connection:
    conn: pg8000.dbapi.Connection = connector.connect(
        instance_connection_name,
        "pg8000",
        user=db_user,
        password=db_pass,
        db=db_name,
        ip_type=ip_type,
    )
    return conn

def load_data_from_database():
    load_dotenv()

    instance_connection_name = os.environ["INSTANCE_CONNECTION_NAME"]
    db_user = os.environ["DB_USER"]
    db_pass = os.environ["DB_PASS"]
    db_name = os.environ["DB_NAME"]
    ip_type = IPTypes.PRIVATE if os.environ.get("PRIVATE_IP") else IPTypes.PUBLIC
    
    connector = Connector(refresh_strategy="LAZY")
    conn = None
    
    try:
        conn = get_conn(connector, instance_connection_name, 
                       db_user, db_pass, db_name, ip_type)
        print("Connection established successfully!")

        query = """
        SELECT 
            id, vehicle_id, area, x, y, width,
            heigth,
            date_time, frame_id
        FROM traffic_data
        ORDER BY date_time DESC
        """

        df_raw = pd.read_sql(query, conn)
        print(f"Loaded {len(df_raw)} records")
        
        df_raw.to_csv("data/raw_traffic_data.csv", index=False)
        print("Data saved to data/raw_traffic_data.csv")
        return df_raw
    
    except Exception as e:
        print(f"Error occurred: {e}")
        raise
    
    finally:
        if conn is not None:
            conn.close()
        connector.close()
        print("✓ Connection closed")