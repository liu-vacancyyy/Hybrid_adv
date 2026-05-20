import numpy as np
import matplotlib.pyplot as plt


# load result
npos_buf = np.load('./result/npos.npy')
epos_buf = np.load('./result/epos.npy')
altitude_buf = np.load('./result/altitude.npy')

# For tracking config
target_npos_buf = np.load('./result/target_npos.npy')
target_epos_buf = np.load('./result/target_epos.npy')
target_altitude_buf = np.load('./result/target_altitude.npy')

# Plot tracking analysis
t = np.arange(npos_buf.shape[0])
plt.plot(t, npos_buf, color='r', label='actual npos')
plt.plot(t, target_npos_buf, color='b', label='target npos')
plt.legend()
plt.xlabel("time/0.02s")
plt.ylabel("npos/feet")
plt.title('North Position Tracking')
plt.show()

t = np.arange(epos_buf.shape[0])
plt.plot(t, epos_buf, color='r', label='actual epos')
plt.plot(t, target_epos_buf, color='b', label='target epos')
plt.legend()
plt.xlabel("time/0.02s")
plt.ylabel("epos/feet")
plt.title('East Position Tracking')
plt.show()

t = np.arange(altitude_buf.shape[0])
plt.plot(t, altitude_buf, color='r', label='actual altitude')
plt.plot(t, target_altitude_buf, color='b', label='target altitude')
plt.legend()
plt.xlabel("time/0.02s")
plt.ylabel("altitude/feet")
plt.title('Altitude Tracking')
plt.show()
