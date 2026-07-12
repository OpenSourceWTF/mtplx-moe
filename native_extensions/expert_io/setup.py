from setuptools import setup

from mlx import extension


if __name__ == "__main__":
    setup(
        name="mtplx_native_expert_io",
        version="0.1.0",
        description="Bounded positional expert reads for MTPLX.",
        ext_modules=[extension.CMakeExtension("mtplx_native_expert_io._ext")],
        cmdclass={"build_ext": extension.CMakeBuild},
        packages=["mtplx_native_expert_io"],
        package_data={"mtplx_native_expert_io": ["*.so", "*.dylib"]},
        zip_safe=False,
        python_requires=">=3.11",
    )
