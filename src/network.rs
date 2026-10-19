use std::{sync::mpsc, thread};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use tokio::net::TcpListener;
use tokio::sync::oneshot;

use crate::tcp::{DEFAULT_MAX_MESSAGE_SIZE, recv_message, send_message};

#[pyclass]
pub struct CoreNode {
    node_id: String,
    port: u16,

    // 用于通知后台关闭
    stop_tx: Option<oneshot::Sender<()>>,

    // Rust 后台线程
    worker: Option<thread::JoinHandle<()>>,
}

#[pymethods]
impl CoreNode {
    #[new]
    pub fn new(node_id: String, port: u16) -> Self {
        Self {
            node_id,
            port,
            stop_tx: None,
            worker: None,
        }
    }

    pub fn start(&mut self, py: Python<'_>) -> PyResult<()> {
        if self.worker.is_some() {
            return Err(PyRuntimeError::new_err("Node already started"));
        }

        let (stop_tx, mut stop_rx) = oneshot::channel();
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);

        let node_id = self.node_id.clone();
        let port = self.port;

        // 创建一个真正的操作系统后台线程
        let worker = thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build();

            let runtime = match runtime {
                Ok(rt) => rt,
                Err(err) => {
                    let _ = ready_tx.send(Err(err.to_string()));
                    return;
                }
            };

            runtime.block_on(async move {
                let address = format!("127.0.0.1:{port}");

                let listener = match TcpListener::bind(&address).await {
                    Ok(listener) => listener,
                    Err(err) => {
                        let _ = ready_tx.send(Err(err.to_string()));
                        return;
                    }
                };

                println!("[{node_id}] Listening on {address}");

                // 告诉 Python：监听端口已经绑定成功
                let _ = ready_tx.send(Ok(()));

                loop {
                    tokio::select! {
                        // 收到关闭信号
                        _ = &mut stop_rx => {
                            break;
                        }

                        // 接受新的 TCP 连接
                        result = listener.accept() => {
                            match result {
                                Ok((mut stream, _)) => {
                                    tokio::spawn(async move {
                                        let result = async {
                                            let data = recv_message(
                                                &mut stream,
                                                DEFAULT_MAX_MESSAGE_SIZE,
                                            ).await?;

                                            // 暂时原样回复收到的数据
                                            send_message(
                                                &mut stream,
                                                &data,
                                                DEFAULT_MAX_MESSAGE_SIZE,
                                            ).await
                                        }.await;

                                        if let Err(err) = result {
                                            eprintln!("Connection error: {err}");
                                        }
                                    });
                                }

                                Err(err) => {
                                    eprintln!("Accept error: {err}");
                                    break;
                                }
                            }
                        }
                    }
                }

                println!("[{node_id}] Stopped");
            });
        });

        // 等待后台完成初始化，但不等待它退出
        let status = py.detach(move || ready_rx.recv());

        match status {
            Ok(Ok(())) => {
                self.stop_tx = Some(stop_tx);
                self.worker = Some(worker);
                Ok(())
            }
            other => {
                let _ = stop_tx.send(());
                let _ = py.detach(move || worker.join());
                Err(PyRuntimeError::new_err(format!(
                    "Failed to start node: {other:?}"
                )))
            }
        }
    }

    pub fn close(&mut self, py: Python<'_>) -> PyResult<()> {
        if let Some(tx) = self.stop_tx.take() {
            let _ = tx.send(());
        }

        if let Some(worker) = self.worker.take() {
            py.detach(move || worker.join())
                .map_err(|_| PyRuntimeError::new_err("Network worker panicked"))?;
        }

        Ok(())
    }
}
