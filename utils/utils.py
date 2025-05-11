from datetime import datetime
def name_with_datetime(prefix='run'):
    now = datetime.now()
    return prefix + '_' + now.strftime("%Y%m%d_%H%M%S")