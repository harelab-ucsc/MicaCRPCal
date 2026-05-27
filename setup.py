from setuptools import find_packages, setup

package_name = 'mica_crp_cal'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    package_data={
        package_name: ['data/*.csv'],
    },
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mwmaster',
    maintainer_email='mwmaster@ucsc.edu',
    description='Pre-flight MicaSense CRP panel scan node.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'panel_scan = mica_crp_cal.panel_scan_node:main',
            'auto_cal   = mica_crp_cal.auto_cal_node:main',
        ],
    },
)
