from google.cloud.sql.connector import Connector, IPTypes
import pg8000

import os
from dotenv import load_dotenv
import sqlalchemy

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

def connect_with_connector() -> sqlalchemy.engine.base.Engine:
    """
    Initializes a connection pool for a Cloud SQL instance of Postgres.

    Uses the Cloud SQL Python Connector package.
    """
    # Note: Saving credentials in environment variables is convenient, but not
    # secure - consider a more secure solution such as
    # Cloud Secret Manager (https://cloud.google.com/secret-manager) to help
    # keep secrets safe.
    load_dotenv()
    instance_connection_name = os.environ["INSTANCE_CONNECTION_NAME"]  # e.g. 'project:region:instance'
    db_user = os.environ["DB_USER"]  # e.g. 'my-db-user'
    db_pass = os.environ["DB_PASS"]  # e.g. 'my-db-password'
    db_name = os.environ["DB_NAME"]  # e.g. 'my-database'

    ip_type = IPTypes.PRIVATE if os.environ.get("PRIVATE_IP") else IPTypes.PUBLIC

    # initialize Cloud SQL Python Connector object
    connector = Connector(refresh_strategy="LAZY")

    def getconn() -> pg8000.dbapi.Connection:
        conn: pg8000.dbapi.Connection = connector.connect(
            instance_connection_name,
            "pg8000",
            user=db_user,
            password=db_pass,
            db=db_name,
            ip_type=ip_type,
        )
        return conn

    # The Cloud SQL Python Connector can be used with SQLAlchemy
    # using the 'creator' argument to 'create_engine'
    engine = create_engine(
        "postgresql+pg8000://",
        creator=getconn,
        # ...
    )
    return engine

def create_engine_session(engine):
    session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    return session

def saveData(engine, frame_id, vehicle_id, area, x, y, width, height, date_time):
    with engine.begin() as c:
        c.execute(text("""
        INSERT INTO traffic_data(frame_id, vehicle_id, area, x, y, width, heigth, date_time)
        VALUES (:frame_id, :vehicle_id, :area, :x, :y, :width, :heigth, :date_time);
                  """), {"frame_id": frame_id, "vehicle_id": vehicle_id, "area": area, "x": x, "y": y, "width": width, "heigth": height, "date_time": date_time})


def t():
    return "t"