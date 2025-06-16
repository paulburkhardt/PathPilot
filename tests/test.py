import rerun as rr
import numpy as np
import time

RERUN_PORT = 9876
rr.connect_grpc(f"rerun+http://127.0.0.1:9876/proxy")
rr.init("rerun_example", default_enabled=True, spawn=True)


# Log a simple 3D point cloud
points = np.random.rand(100, 3)
rr.log("my_points", rr.Points3D(points))
time.sleep(31)


