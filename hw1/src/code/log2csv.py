import numpy as np

src_path_log = './../data/log/robotdata1.log'

data = []

logfile = open(src_path_log, 'r')

for time_idx, line in enumerate(logfile):
    # Read a single 'line' from the log file (can be either odometry or laser measurement)
    # L : laser scan measurement, O : odometry measurement
    meas_type = line[0]

    # convert measurement values from string to double
    meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=' ')
    
    if (meas_type == "L"):
        # 180 range measurement values from single laser scan
        ranges = meas_vals[6:-1]
        data.append(ranges)

data = np.array(data)
print(data.shape)

np.savetxt('log.csv', data, fmt="%d", delimiter=',')