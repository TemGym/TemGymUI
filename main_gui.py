import temgym_core.components as comp
import temgym_core.source as sources
from temgym_ui.run import temgym


model = (
    sources.ParallelBeam(0., 0.01),
    comp.Lens(0.5, 0.05),
    comp.Detector(1., (0.001,) * 2, (128, 128)),
)

temgym(model)
