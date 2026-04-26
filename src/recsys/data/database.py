import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import BigInteger, Column, Float, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

load_dotenv()

# Load connection string from environment variable
DATABASE_URL = os.getenv("NEON_DB_URL")

if not DATABASE_URL:
    raise ValueError("NEON_DB_URL environment variable is not set")

Base = declarative_base()

class Product(Base):
    __tablename__ = 'products'
    _id = Column(BigInteger, primary_key=True, autoincrement=True)
    itemId = Column(Integer, index=True)
    pricelog2 = Column(Float)
    product_name_tokens = Column(String)

class ProductCategory(Base):
    __tablename__ = 'product_categories'
    _id = Column(BigInteger, primary_key=True, autoincrement=True)
    itemId = Column(Integer, index=True)
    categoryId = Column(Integer)

class UserView(Base):
    __tablename__ = 'user_views'
    _id = Column(BigInteger, primary_key=True, autoincrement=True)
    sessionId = Column(String)
    userId = Column(String)
    itemId = Column(Integer, index=True)
    timeframe = Column(BigInteger)
    eventdate = Column(String)

def init_db():
    print("Forcing schema reset (dropping and creating tables)...")
    engine = create_engine(DATABASE_URL)
    # Forced removal of current schema
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    return engine

def ingest_data(engine):
    print("Ingesting products.csv (raw)...")
    products_df = pd.read_csv('data/raw/products.csv', sep=';')
    # Map dot-notation column to SQL-safe name
    products_df = products_df.rename(
        columns={"product.name.tokens": "product_name_tokens"}
    )
    # _id is autoincremented by DB, so we don't provide it in the dataframe
    products_df.to_sql(
        "products", engine, if_exists="append", index=False, chunksize=1000
    )

    print("Ingesting product-categories.csv (raw)...")
    categories_df = pd.read_csv("data/raw/product-categories.csv", sep=";")
    categories_df.to_sql(
        "product_categories", engine, if_exists="append", index=False, chunksize=1000
    )
    
    print("Ingestion complete.")

if __name__ == "__main__":
    engine = init_db()
    ingest_data(engine)
    
    # Verification
    Session = sessionmaker(bind=engine)
    session = Session()
    product_count = session.query(Product).count()
    category_count = session.query(ProductCategory).count()
    print(
        f"Verified: {product_count} products and {category_count} categories "
        "uploaded to Neon."
    )
    session.close()
