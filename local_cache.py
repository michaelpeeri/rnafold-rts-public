# Provide persistent string->string mapping (backed by sqlite file containing a single table). Multiple mappings can be created by passing different identifiers to the constructor.
from builtins import range
from builtins import object
import sys
from sqlalchemy import create_engine, Table, Column, MetaData, sql, String
from sqlalchemy.dialects.sqlite import INTEGER, FLOAT, VARCHAR
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
import config


#---- Configuration -------------------------------------------
_debug = False
#--------------------------------------------------------------


def get_db(filename):
    db = create_engine(config.make_sqlite_host_connection(filename), encoding='ascii', echo=_debug)
    connection = db.connect()
    return db, connection

#--------------------------------------------------------------
# ORM schema definition

md = MetaData()

strings_map = Table("strings_map", md,
                    Column("key", String, primary_key=True),
                    Column("value", String))

Base = declarative_base()


class StringsMap(Base):
    __tablename__ = "strings_map"
    key = Column(String, primary_key=True)
    value = Column(String)

#--------------------------------------------------------------

# Note - keys are assumed to be unique (within the scope of the store). This is enforced by the PK constraint.
class LocalStringsCache(object):
    def __init__(self, storeName):
        self._db, self._connection = get_db(storeName)
        self._storeName = storeName
        md.create_all(self._db)
        self._session = sessionmaker(bind=self._db)()

    def get_value(self, key):
        results = self._connection.execute( sql.select( (strings_map.c.value, )).select_from(strings_map).where(
            sql.and_(
                strings_map.c.key == key
            )
        ) )  # Note: order_by not needed, because id is used

        ret = results.fetchall()
        if( len(ret) < 1 ):
            return None
        return ret[0][0]

    def all_matching_values_source(self, prefix ):
        results = self._connection.execute( sql.select( (strings_map.c.key, strings_map.c.value, )).select_from(strings_map).where(
            strings_map.c.key.startswith( prefix )
        ) )  # Note: order_by not needed, because id is used

        for u, v in results:
            yield (u,v)
        

    def insert_value(self, key, value):
        sm = StringsMap( key=key,
                         value=value )
        self._session.add(sm)
        self._session.commit() # may throw IntegrityError

    def update_value(self, key, newValue):
        row = self._session.query(StringsMap).filter_by( key=key ).first()
        if row is None:
            raise Exception("Couldn't find StringsMap row matching key={}".format( key ) )
        row.value = newValue
        self._session.commit()


def testAll():
    from nucleic_compress import randseq
    sc1 = LocalStringsCache("test_sc1")
    #sc1.insert_value("test1", "test1")
    #sc1.insert_value("test2", "test2")
    #sc1.update_value( "test1", "----1----" )
    #sc1.update_value( "test1", "----2----" )
    #sc1.update_value( "test1", "----3----" )
    sc1.update_value( "test1", "test1" )
    #sc1.update_value( "test3", "test3" )

    #for i in range(10000):
    #    sc1.insert_value("x.{}".format(i), randseq(1000) )
    for i in range(1000):
        sc1.update_value("x.{}".format(i), randseq(1000) )

    for u, v in sc1.all_matching_values_source( "x." ):
        print(u,v[:10])
        
    print(sc1.get_value("test1"))
    print(sc1.get_value("test1X"))
    assert( sc1.get_value("test1") == "test1" )
    return 0


if __name__=="__main__":
    import sys
    sys.exit(testAll())
