import queue
import pg8000
from datetime import datetime
import time
import os
import dotenv
import smtplib, ssl
from google.cloud.sql.connector import Connector, IPTypes

dotenv.load_dotenv()
instance_connection_name = os.environ["INSTANCE_CONNECTION_NAME"]  # e.g. 'project:region:instance'
db_user = os.environ["DB_USER"]  # e.g. 'my-db-user'
db_pass = os.environ["DB_PASS"]  # e.g. 'my-db-password'
db_name = os.environ["DB_NAME"]  # e.g. 'my-database'

ip_type = IPTypes.PRIVATE if os.environ.get("PRIVATE_IP") else IPTypes.PUBLIC

connector = Connector(refresh_strategy="LAZY")

log_queue = queue.Queue(maxsize=1000000)

# Email information
gmail_smtp_server = os.environ["GMAIL_SMTP_SERVER"]
port = int(os.environ["SSL_PORT"])
password = os.environ["GMAIL_SEND_MAIL_PASSWORD"]
sender_email = os.environ["GMAIL_SENDER_EMAIL"]
receiver_email = "fiorenzo.luethi@me.com"
message = """\
Subject: Error saving data to database

Error occurred while saving data to the database. Please check the system."""

ssl_context = ssl.create_default_context()

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

conn = get_conn()
cursor = conn.cursor()

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
                manage_data_insertion(cursor, conn, batch)
                batch.clear()
                last_flush = time.time()
            continue

        if item is StopIteration:
            # Shutdown signal: flush remaining and exit
            if batch:
                manage_data_insertion(cursor, conn, batch)
            break

        # Normal item: add to batch
        batch.append(item)

        if len(batch) >= BATCH_SIZE:
            manage_data_insertion(cursor, conn, batch)
            batch.clear()
            last_flush = time.time()

    cursor.close()
    conn.close()
def manage_data_insertion(batch):
    MAX_RETRIES = 5
    retries = 0
    while retries < MAX_RETRIES:
        try:
            insert_batch(batch)
            return
        except pg8000.InterfaceError as e:
            retries += 1
            print(f"Network error, reconnecting ({retries}/{MAX_RETRIES})")

            try:
                cursor.close()
                conn.close()
            except Exception:
                pass
            time.sleep(1)
            conn = get_conn()
            cursor = conn.cursor()
    #Backup the existing queue to a file to minimize data loss
    unsaved_data = list(log_queue.queue)
    unsaved_data.append(batch)
    with open('data_when_network_error.csv','a') as fd:
        fd.write(unsaved_data)
    # Clears both the batch and the queue
    batch.clear()
    log_queue.queue.clear()
    last_flush = time.time()
    # Send email to Fiorenzo
    with smtplib.SMTP_SSL(gmail_smtp_server, port, context=ssl_context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)

    raise Exception(f"Network error: Failed to insert batch after {MAX_RETRIES} retries. Saving remaining data to queue")


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
        INSERT INTO traffic_data (
            vehicle_id, area, x, y, width, heigth, date_time, frame_id
        ) VALUES {values_sql}
    """

    # Flatten the list of tuples into one big parameter list
    params = []
    for row in normalized:
        params.extend(row)

    cursor.execute(sql, params)
    conn.commit()