import osmium
import psycopg2
import json

from sijapi import DB_USER, DB_PASS, DB_HOST, DB, DATA_DIR

OSM_DATA_PATH = DATA_DIR / "north-america-latest.osm.pbf"

class OSMHandler(osmium.SimpleHandler):
    def __init__(self, conn):
        osmium.SimpleHandler.__init__(self)
        self.conn = conn

    def node(self, n):
        tags = {tag.k: tag.v for tag in n.tags}
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO nodes (id, location, tags) 
            VALUES (%s, ST_SetSRID(ST_MAKEPOINT(%s, %s),4326), %s)
            """,
            (n.id, n.location.lon, n.location.lat, json.dumps(tags)))
        self.conn.commit()

    def way(self, w):
        nodes = [(node.lon, node.lat) for node in w.nodes]
        tags = {tag.k: tag.v for tag in w.tags}
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO ways (id, nodes, tags) 
            VALUES (%s, %s, %s)
            """,
            (w.id, json.dumps(nodes), json.dumps(tags)))
        self.conn.commit()

    def relation(self, r):
        members = [{"type": m.type, "ref": m.ref, "role": m.role} for m in r.members]
        tags = {tag.k: tag.v for tag in r.tags}
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO relations (id, members, tags) 
            VALUES (%s, %s, %s)
            """,
            (r.id, json.dumps(members), json.dumps(tags)))
        self.conn.commit()

def main():
    conn = psycopg2.connect(user=DB_USER, password=DB_PASS, dbname=DB, host=DB_HOST)
    cur = conn.cursor()
    
    # Drop existing tables if they exist
    cur.execute("DROP TABLE IF EXISTS nodes")
    cur.execute("DROP TABLE IF EXISTS ways")
    cur.execute("DROP TABLE IF EXISTS relations")
    
    # Create tables for nodes, ways, and relations
    cur.execute("""
        CREATE TABLE nodes (
            id bigint PRIMARY KEY,
            location geography(POINT, 4326),
            tags jsonb
        )
    """)
    
    cur.execute("""
        CREATE TABLE ways (
            id bigint PRIMARY KEY,
            nodes jsonb,
            tags jsonb
        )
    """)
    
    cur.execute("""
        CREATE TABLE relations (
            id bigint PRIMARY KEY,
            members jsonb,
            tags jsonb
        )
    """)
    
    conn.commit()

    handler = OSMHandler(conn)
    handler.apply_file(str(OSM_DATA_PATH))

    cur.close()
    conn.close()

if __name__ == "__main__":
    main()