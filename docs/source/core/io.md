# Input/Output (`deepmr.io`)

```{eval-rst}
.. automodule:: deepmr.io
```

## Image I/O routines
```{eval-rst}
.. autosummary::
	:toctree: generated
	:nosignatures:
	
	deepmr.io.read_image
	deepmr.io.write_image
```

## K-Space I/O routines
```{eval-rst}
.. autosummary::
	:toctree: generated
	:nosignatures:
	
	deepmr.io.read_rawdata
```

## Header I/O routines
```{eval-rst}
.. autosummary::
	:toctree: generated
	:nosignatures:
	
	deepmr.io.read_acquisition_header
    deepmr.io.write_acquisition_header
```

## Generic I/O routines

DeepMR also contains generic I/O routines that can be used to write custom I/O functions for other formats. As a general rule, functions should have these signatures:

```python
# Custom reading routines
data, head = read_custom_rawdata(filepath, *args, **kwargs) # k-space
image, head = read_custom_image(filepath, *args, **kwargs) # image
head = read_custom_acqheader(filepath, *args, **kwargs) # header

# Custom writing routines
write_custom_rawdata(filepath, data, *args, head=None **kwargs) # k-space
write_custom_image(filepath, image, *args, head=None, **kwargs) # image
write_custom_acqheader(filepath, head, *args, **kwargs) # header
```



```{eval-rst}
.. autosummary::
	:toctree: generated
	:nosignatures:
	
	deepmr.io.read_matfile
	deepmr.io.read_hdf5
	deepmr.io.write_hdf5
	deepmr.io.Header
```
