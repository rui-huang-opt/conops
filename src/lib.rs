use pyo3::prelude::*;

pub mod exchange;
pub mod message;
pub mod network;
pub mod tcp;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<network::CoreNetwork>()?;
    Ok(())
}
