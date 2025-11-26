import queue
import pg8000
from datetime import datetime
import time
import os
import dotenv
from google.cloud.sql.connector import Connector, IPTypes

dotenv.load_dotenv()
instance_connection_name = os.environ["INSTANCE_CONNECTION_NAME"]  # e.g. 'project:region:instance'
db_user = os.environ["DB_USER"]  # e.g. 'my-db-user'
db_pass = os.environ["DB_PASS"]  # e.g. 'my-db-password'
db_name = os.environ["DB_NAME"]  # e.g. 'my-database'

ip_type = IPTypes.PRIVATE if os.environ.get("PRIVATE_IP") else IPTypes.PUBLIC

connector = Connector(refresh_strategy="LAZY")

log_queue = queue.Queue(maxsize=100000000)

def get_conn() -> pg8000.dbapi.Connection:
    conn: pg8000.dbapi.Connection = connector.connect(
        instance_connection_name,
        "pg8000",
        user=db_user,
        password=db_pass,
        db=db_name,
        ip_type=ip_type,
    )
    return conn

def db_writer_thread():
    conn = get_conn()
    cursor = conn.cursor()

    batch = []
    BATCH_SIZE = 200         # Number of rows per insert
    BATCH_TIMEOUT = 0.2     # Max seconds to wait before flushing

    last_flush = time.time()

    while True:
        try:
            item = log_queue.get(timeout=0.05)
        except queue.Empty:
            item = None

        if item is None:
            # No new item; check if we should flush existing batch
            if batch and (time.time() - last_flush) >= BATCH_TIMEOUT:
                insert_batch(cursor, conn, batch)
                batch.clear()
                last_flush = time.time()
            continue

        if item is StopIteration:
            # Shutdown signal: flush remaining and exit
            if batch:
                insert_batch(cursor, conn, batch)
            break

        # Normal item: add to batch
        batch.append(item)

        if len(batch) >= BATCH_SIZE:
            insert_batch(cursor, conn, batch)
            batch.clear()
            last_flush = time.time()

    cursor.close()
    conn.close()

def insert_batch(cursor, conn, batch):
    """
    batch: list of tuples
        (vehicle_id, area, x, y, width, heigth, date_time, frame_id)
    """
    if not batch:
        return

    normalized = []
    for vehicle_id, area, x, y, width, heigth, date_time, frame_id in batch:
        # Normalize timestamp
        if isinstance(date_time, (int, float)):
            date_time = datetime.fromtimestamp(date_time)

        normalized.append((
            int(vehicle_id),
            float(area),
            int(x),
            int(y),
            float(width),
            float(heigth),
            date_time,
            int(frame_id),
        ))

    # Build a single multi-VALUES INSERT
    values_template = "(" + ",".join(["%s"] * 8) + ")"
    values_sql = ",".join([values_template] * len(normalized))

    sql = f"""
        INSERT INTO test (
            vehicle_id, area, x, y, width, heigth, date_time, frame_id
        ) VALUES {values_sql}
    """

    # Flatten the list of tuples into one big parameter list
    params = []
    for row in normalized:
        params.extend(row)

    cursor.execute(sql, params)
    conn.commit()