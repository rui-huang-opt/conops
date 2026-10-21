use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

pub mod discovery;
pub mod exchange;
pub mod message;
pub mod network;
pub mod tcp;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::try_init().map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

    m.add_class::<network::CoreNetwork>()?;
    Ok(())
}
