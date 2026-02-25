# GPZ

GPZ is a high-performance, error-bounded lossy compressor designed specifically for large-scale particle data on modern GPUs. GPZ employs a novel four-stage parallel pipeline that synergistically balances high compression efficiency with the architectural demands of massively parallel hardware.

## Environment Requirements

- Linux OS amd64 with NVIDIA GPUs (Tested on RTX 4090 and H100, which means compute compatibility 8.0+)
- CMake >= 3.22
- Cuda Toolkit >= 12.0
- GCC >= 12.0

## Compile

- enter root folder
- `mkdir build`
- if using cc 8.9(e.g. RTX 4090) `cmake -DCMAKE_BUILD_TYPE=Release -DREGISTER_LIMIT=168 -S . -B build`
- if using cc 9.0(e.g. H100) `cmake -DCMAKE_BUILD_TYPE=Release -DREGISTER_LIMIT=128 -S . -B build`
- `cd build`
- `make -j 30`

After compilation, the executable `gpz` will be generated in the `build/` folder.

## Usage

compression:

```bash
<program> -x -i <input_file> -o <output_file> -n <num_points> -p <dim> -f <bits> -m <error_mode> -e <error_bound>
```

decompression:

```bash
<program> -d -o <compressed_file> -r <reconstructed_file> [-v <original_file>]
```

### Example Usage

```shell
# compress
./gpz -x -i merged_xyz.f32 -o ./merged_xyz.f32.gpz.compressed -f 32 -n 143735721 -p 3 -m rel -e 1e-4

# decompress
./gpz -d -o merged_xyz.f32.gpz.compressed -r merged_xyz.f32.gpz.reconstructed

# decompress with verification
./gpz -d -o merged_xyz.f32.gpz.compressed -r merged_xyz.f32.gpz.reconstructed -v merged_xyz.f32
```

### Compression Parameters

Note: The input file should be a merged file with x, y, z coordinates (for example, x1,y1,z1...xn,yn,zn)

| Option         | Type       | Description                                                                             |
| -------------- | ---------- | --------------------------------------------------------------------------------------- |
| `-x`         | `flag`   | **(Required)** Enable compression mode                                            |
| `-i <file>`  | `string` | **(Required)** Input file to compress                                             |
| `-o <file>`  | `string` | **(Required)** Output compressed file                                             |
| `-p <dim>`   | `int`    | **(Optional)** Number of dimensions, only `2` or `3` allowed (default: `3`) |
| `-n <num>`   | `int`    | **(Required)**  Number of points                                                  |
| `-m <mode>`  | `string` | **(Required)** Error mode, e.g., `rel`, `abs`                                 |
| `-f <bits>`  | `int`    | **(Required)** Float precision bits, must be `32` or `64`                     |
| `-e <value>` | `float`  | **(Required)** Error bound, e.g. 1e-4                                             |
| `-s <size>`  | `int`    | **(Optional)** Segment size, auto tune if not specified                           |

---

### Decompression Parameters

| Option        | Type       | Description                                              |
| ------------- | ---------- | -------------------------------------------------------- |
| `-d`        | `flag`   | **(Required)** Enable decompression mode           |
| `-o <file>` | `string` | **(Required)** Output decompressed file            |
| `-r <file>` | `string` | **(Optional)** Output recontracted (refitted) file |
| `-v <file>` | `string` | **(Optional)** Original file for verification      |

---

<details>
<summary><strong> Optional: nvcomp Support</strong></summary>

GPZ optionally supports [nvcomp](https://developer.nvidia.com/nvcomp) for additional lossless compression on top of the lossy-compressed output. This feature is **disabled by default** and must be explicitly enabled at compile time since it's mainly for testing purpose. We tested on nvcomp version 3.0.5. For others, you may need to modify the nvcomp API calls in the code accordingly.

## Using with nvcomp

### Compile

Pass `-DNVCOMP_DIR=<path>` to CMake, pointing to your nvcomp installation directory (which should contain `include/` and `lib/` subdirectories):

```bash
# Example
cmake -DCMAKE_BUILD_TYPE=Release -DREGISTER_LIMIT=168 -DNVCOMP_DIR=/path/to/nvcomp -S . -B build
```

### Additional Parameters (Compression)

| Option        | Type       | Description                                                                                                                                      |
| ------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `-t <type>` | `string` | **(Optional)** nvcomp compression type, e.g., `lz4`, `bitcomp`, `snappy`, `ans`, `cascaded`, `gdeflate`, `deflate`, `zstd` |

### Additional Parameters (Decompression)

When `-t` is used, the following parameters are also **required**, since metadata cannot be read directly from the nvcomp-wrapped compressed file:

| Option         | Type       | Description                                                                                                                                     |
| -------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `-t <type>`  | `string` | **(Required)** nvcomp compression type used during compression                                                                            |
| `-n <num>`   | `int`    | **(Required)** Number of points                                                                                                           |
| `-e <value>` | `float`  | **(Required)** ABS Error bound (Note: if you used relative error mode during compression, you should convert it to absolute error bound.) |
| `-f <bits>`  | `int`    | **(Required)** Float precision bits (`32` or `64`)                                                                                    |
| `-p <dim>`   | `int`    | **(Optional)** Number of dimensions (default: `3`)                                                                                      |

### Example Usage with nvcomp

```shell
# compress with bitcomp
./gpz -x -i merged_xyz.f32 -o ./merged_xyz.f32.gpz.compressed -f 32 -n 83953207 -p 3 -m rel -e 1e-4 -t bitcomp

# decompress (must provide -t, -n, -e, -f)
./gpz -d -o merged_xyz.f32.gpz.compressed -r merged_xyz.f32.gpz.reconstructed -t bitcomp -p 3 -f 32 -n 83953207 -e 0.2999999821

# decompress with verification
./gpz -d -o merged_xyz.f32.gpz.compressed -r merged_xyz.f32.gpz.reconstructed -v merged_xyz.f32 -t bitcomp -p 3 -f 32 -n 83953207 -e 1e-4
```

</details>

## Sample Datasets

Dataset could be downloaded from https://sdrbench.github.io/

For example:
The HACC dataset with 1,073,726,487 points; You may need to combined the x, y, z coordinates into a single file.

## Citation

If you find GPZ useful in your research, please consider citing:

__[ICS'26]__ GPZ: GPU-Accelerated Lossy Compressor for Particle Data

```bibtex
@inproceedings{li2026gpz,
    title={GPZ: GPU-Accelerated Lossy Compressor for Particle Data},
    author={Li, Ruoyu and Huang, Yafan and Zhang, Longtao and Yang, Zhuoxun and Di, Sheng and Zhang, Boyuan and Huang, Jiajun and Liu, Jinyang and Tian, Jiannan and Li, Guanpeng and Song, Fengguang and Guo, Hanqi and Cappello, Franck and Zhao, Kai},
    booktitle={Proceedings of the ACM International Conference on Supercomputing},
    year={2026}
}
```
