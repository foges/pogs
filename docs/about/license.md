# License

POGS is licensed under the **Apache License, Version 2.0**.

---

## Apache License 2.0

```
Copyright 2014-2026 Chris Fougner and Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

## What This Means

### You Are Free To:

- ✅ **Use** POGS for any purpose (commercial or non-commercial)
- ✅ **Modify** the source code
- ✅ **Distribute** copies of POGS
- ✅ **Distribute** modified versions
- ✅ **Use** POGS in proprietary software

### Requirements:

- 📋 **License notice**: Include a copy of the license
- 📋 **Notice of changes**: Document significant modifications
- 📋 **Copyright notice**: Preserve copyright notices
- 🚫 **No trademark use**: Don't use project trademarks without permission

### No Warranty:

- ⚠️ Software provided "AS IS" without warranties
- ⚠️ No liability for damages from use

---

## Third-Party Dependencies

POGS uses the following libraries:

### BLAS/LAPACK

- **License**: BSD-style
- **Usage**: Linear algebra operations
- **Platforms**:
  - macOS: Accelerate framework (Apple)
  - Linux: OpenBLAS or ATLAS
  - Windows: Intel MKL or OpenBLAS

### GSL (Optional)

- **License**: GPL v3
- **Usage**: Some mathematical functions
- **Note**: POGS includes GSL-style wrappers, not the full GSL library

### CUDA (Optional)

- **License**: NVIDIA CUDA EULA
- **Usage**: GPU acceleration
- **Required**: Only for GPU builds

---

## Contributing

By contributing to POGS, you agree that your contributions will be licensed under the Apache License 2.0.

See [Contributing Guide](../developer/contributing.md) for details.

---

## Citations

If you use POGS in academic work, please cite:

```bibtex
@article{fougner2016pogs,
  title={Parameter selection and preconditioning for a graph form solver},
  author={Fougner, Chris and Boyd, Stephen},
  journal={Optimization and Engineering},
  year={2016},
  publisher={Springer}
}
```

---

## Full License Text

The complete Apache License 2.0 text is available at:

[https://www.apache.org/licenses/LICENSE-2.0](https://www.apache.org/licenses/LICENSE-2.0)

---

## Contact

For licensing questions, please open an issue on GitHub:

[https://github.com/foges/pogs/issues](https://github.com/foges/pogs/issues)
