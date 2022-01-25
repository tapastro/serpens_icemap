from dustmaps.config import config
import dustmaps.sfd

config.reset()
config['data_dir'] = 'dustmaps/'
dustmaps.sfd.fetch()
