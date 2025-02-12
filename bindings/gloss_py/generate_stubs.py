"""Generate type stubs for gloss"""
import os
from typing import Any, Dict, Optional
import inspect
import gloss

def parse_signature(obj: Any) -> Optional[str]:
    """Extract signature from docstring."""
    if not (hasattr(obj, '__doc__') and obj.__doc__):
        return None
    doc = obj.__doc__.split('--')[0].strip()
    if '(' not in doc:
        return None
    # print(f"PARSING: {doc}")
    return doc

def parse_params(sig: str) -> Dict[str, str]:
    """Extract parameter types."""
    if '(' not in sig or ')' not in sig:
        return {}
    params = {}
    param_str = sig[sig.find('(')+1:sig.find(')')].strip()
    if not param_str:
        return params

    for p in param_str.split(','):
        p = p.strip()
        if ':' not in p:
            continue
        name, type_hint = p.split(':', 1)
        name = name.strip()
        if '=' in type_hint:
            type_hint = type_hint.split('=')[0]
        params[name] = type_hint.strip()

    # print(f"PARAMS: {params}")
    return params

def generate_stub(module, f, is_types_module=False):
    """Generate stubs for a module."""
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj):
            # Special handling for enums in types module
            if is_types_module:
                # Get all non-magic methods/attributes
                enum_variants = [attr for attr in dir(obj) if not attr.startswith('_')]
                if enum_variants:
                    f.write(f"class {name}:\n")
                    for variant in enum_variants:
                        f.write(f"    {variant}: int = ...\n")
                    f.write("\n")
                    continue

            f.write(f"class {name}:\n")

            # Handle methods
            methods = inspect.getmembers(obj, predicate=inspect.isfunction)
            if not methods and not any(m for m, _ in inspect.getmembers(obj) if callable(getattr(obj, m, None)) and not m.startswith('__')):
                f.write("    pass\n")
            else:
                for method_name, method in inspect.getmembers(obj):
                    if not callable(getattr(obj, method_name, None)) or method_name.startswith('__'):
                        continue

                    if method_name == '__init__':
                        sig = parse_signature(obj)  # Use class docstring for __init__
                    else:
                        sig = parse_signature(method)

                    if sig:
                        params = parse_params(sig)
                        param_list = []
                        if 'self' in params or method_name != '__init__':
                            param_list.append('self')
                        param_list.extend(f"{k}: {v}" for k, v in params.items() if k != 'self')
                        ret_type = 'None'
                        if '->' in sig:
                            ret_type = sig.split('->')[1].strip()
                        f.write(f"    def {method_name}({', '.join(param_list)}) -> {ret_type}: ...\n")
                    else:
                        f.write(f"    def {method_name}(*args: Any, **kwargs: Any) -> Any: ...\n")
            f.write("\n")

def generate_stubs():
    """Generate all stubs."""
    os.makedirs('gloss', exist_ok=True)

    # Common imports for all stub files
    common_imports = [
        "from __future__ import annotations",
        "from typing import Any, Optional, List, Type, TypeVar, Tuple",
        "import numpy as np",
        "from numpy.typing import NDArray",
        "",
        "T = TypeVar('T')",
        ""
    ]

    # Module-specific imports
    module_imports = {
        '__init__': [
            "from gloss.types import IndirRemovalPolicy, SplatType",
            "from gloss.log import LogLevel, LogLevelCaps",
            "from gloss.components import Colors, DiffuseImg, Edges, Faces, Normals, Tangents, UVs, Verts, VisLines, VisMesh, VisPoints, ModelMatrix",
            "from gloss.builders import EntityBuilder",
        ],
        'types': [],
        'log': [],
        'components': [],
        'builders': []
    }

    for submod in ['types', 'log', 'components', 'builders', '__init__']:
        try:
            with open(f'gloss/{submod}.pyi', 'w') as f:
                f.write('\n'.join(common_imports))
                if submod in module_imports:
                    f.write('\n'.join(module_imports[submod]) + '\n\n')
                if submod == '__init__':
                    generate_stub(gloss, f)
                else:
                    generate_stub(getattr(gloss, submod), f, is_types_module=(submod == 'types'))
        except AttributeError:
            print(f"Warning: Submodule {submod} not found")

if __name__ == "__main__":
    generate_stubs()
