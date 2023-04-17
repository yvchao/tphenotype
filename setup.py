import os
import re

from Cython.Build import cythonize
from setuptools import Extension, setup

if __name__ == "__main__":
    try:
        PKG = "tphenotype"

        # Configure Cython lexsort extension.
        cython_config = {
            "compiler_directives": {"language_level": "3"},
        }
        sources = [f"src/{PKG}/utils/lexsort.pyx"]
        extension = Extension("lexsort", sources=sources, extra_compile_args=["-std=c++11"])
        extensions = [extension]

        # Configure version.
        def find_version() -> str:
            def read(fname: str) -> str:
                return open(os.path.realpath(os.path.join(os.path.dirname(__file__), fname)), encoding="utf8").read()

            version_file = read(f"src/{PKG}/version.py").split("\n")[0]
            version_re = r"__version__ = \"(?P<version>.+)\""
            version_raw = re.match(version_re, version_file)

            if version_raw is None:
                raise RuntimeError(f"__version__ value not found, check src/{PKG}/version.py")

            version = version_raw.group("version")
            return version

        setup(
            version=find_version(),
            ext_package=f"{PKG}.utils",
            ext_modules=cythonize(extensions, **cython_config),
        )
    except:  # noqa
        print("\n\nAn error occurred while building the project")
        raise
