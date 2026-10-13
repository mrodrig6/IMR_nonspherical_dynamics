"""Field reconstruction and rendering.

:mod:`imrns.postprocess.fields` turns the solver's modal output into
displacement, strain, strain-rate and stress fields and returns arrays.
:mod:`imrns.postprocess.render` draws them.  The split is deliberate: in the
MATLAB original the two were interleaved across 2769 lines, which left the
field reconstruction untestable on its own.
"""

from .fields import (ModalField, build_modal_field, harmonics,
                     rate_of_strain_components, strain_components,
                     stress_components, tensor_scalar)
from .render import FieldSpec, animate, field_slice, plot_frame, snapshot_grid

__all__ = [
    "ModalField",
    "build_modal_field",
    "harmonics",
    "strain_components",
    "rate_of_strain_components",
    "stress_components",
    "tensor_scalar",
    "FieldSpec",
    "field_slice",
    "plot_frame",
    "snapshot_grid",
    "animate",
]
