import logging
from dataclasses import dataclass
# Site configuration file (for server details, etc.)


@dataclass(frozen=True)
class ConfigTermfold:
    # redis
    host = "compute-0-89"
    port = 6381
    db = 2
    password = "termfold"

    # sqlite
    sqlite_base_path = "/tamir1/mich1/termfold/data"
    def make_sqlite_host_connection(filename):
        return 'sqlite:///%s/%s.db' % (sqlite_base_path, filename)

    # mysql
    mysql_host_connection = 'mysql+mysqlconnector://---username---:---password---@---host---:---port---/term?charset=latin1&use_unicode=0'
    
@dataclass(frozen=True)
class Config:
    termfold = ConfigTermfold()
    default  = termfold
config = Config()
    
run_without_mysql_server = False



# data files
base_data_dir    = "/tamir1/mich1/termfold/data"
ensembl_data_dir = "/tamir1/mich1/termfold/data/Ensembl"


# Pushover
# Application token
computation_monitor_app = '---pushover-token---'
# Group key
computation_monitor_group = '---pushover-group---'

# Matlab
MatlabPath = "/usr/local.cc/bin/matlab"
# codonW
codonwBasePath = '/a/home/cc/students/enginer/mich1/src/codonW'
# ENCprime
ENCprimeBasePath = '/a/home/cc/students/enginer/mich1/src/ENCprime-master/bin/'

def process_id():
    import subprocess
    import os
    return "%s_%d" % (subprocess.check_output('hostname')[:-1], os.getpid())
 
# logging
def initLogging():
	logging.basicConfig(filename='process_%s.log' % process_id(), level=logging.INFO)


