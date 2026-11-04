from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'quarry_drone_vlm'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Nikita',
    maintainer_email='nikita@localhost',
    description='Управление БПЛА с помощью VLM Qwen2.5-VL в виртуальном карьере Gazebo Harmonic',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'mission_node = quarry_drone_vlm.mission_node:main',
            'operator_cli = quarry_drone_vlm.operator_cli:main',
        ],
    },
)
