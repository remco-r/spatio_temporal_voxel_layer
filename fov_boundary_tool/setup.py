from setuptools import setup, find_packages

setup(
    name='fov_boundary_tool',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'PyQt5',
        'PyYAML',
        'scipy',
    ],
    entry_points={
        'console_scripts': [
            'fov_boundary_tool=fov_boundary_tool.main:main',
        ],
    },
    description='GUI tool for defining convex polygon FOV blind spot boundaries on lidar range images',
    author='Remco',
    license='proprietary',
)
