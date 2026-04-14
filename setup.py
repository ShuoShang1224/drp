from setuptools import setup, find_packages

setup(
    name="drp",
    version="0.0.0",
    description="Pre-release version of DRP",
    packages=find_packages(include=["drp", "drp.*"]),
    package_data={"drp": ["utils/pcd_cache/franka/*.npy"]},
)
