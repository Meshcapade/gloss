//! Transport layer abstraction for network communication.
//!
//! This module defines traits and implementations for sending/receiving
//! scene synchronization messages over various transport mechanisms.

use super::messages::MessageFrame;
use std::io;
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::sync::{Arc, Mutex};

/// Error type for transport operations
#[derive(Debug)]
pub enum TransportError {
    Io(io::Error),
    Serialization(bincode::error::EncodeError),
    Deserialization(bincode::error::DecodeError),
    ConnectionClosed,
    NotConnected,
    ChannelDisconnected,
}

impl std::fmt::Display for TransportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransportError::Io(e) => write!(f, "IO error: {e}"),
            TransportError::Serialization(e) => write!(f, "Serialization error: {e}"),
            TransportError::Deserialization(e) => write!(f, "Deserialization error: {e}"),
            TransportError::ConnectionClosed => write!(f, "Connection closed"),
            TransportError::NotConnected => write!(f, "Not connected"),
            TransportError::ChannelDisconnected => write!(f, "Channel disconnected"),
        }
    }
}

impl std::error::Error for TransportError {}

impl From<io::Error> for TransportError {
    fn from(e: io::Error) -> Self {
        TransportError::Io(e)
    }
}

impl From<bincode::error::EncodeError> for TransportError {
    fn from(e: bincode::error::EncodeError) -> Self {
        TransportError::Serialization(e)
    }
}

impl From<bincode::error::DecodeError> for TransportError {
    fn from(e: bincode::error::DecodeError) -> Self {
        TransportError::Deserialization(e)
    }
}

/// Trait for network transports
pub trait Transport: Send + Sync {
    /// Send a message
    #[allow(clippy::missing_errors_doc)]
    fn send(&self, frame: &MessageFrame) -> Result<(), TransportError>;

    /// Receive a message (blocking if block=true, non-blocking if block=false)
    #[allow(clippy::missing_errors_doc)]
    fn receive(&self, block: bool) -> Result<Option<MessageFrame>, TransportError>;

    /// Check if connected
    fn is_connected(&self) -> bool;

    /// Connect to peer (no-op for servers until client connects)
    #[allow(clippy::missing_errors_doc)]
    fn connect(&self) -> Result<(), TransportError>;

    /// Close the connection
    fn close(&self);
}

/// Configuration for network transport
#[derive(Debug, Clone)]
pub struct TransportConfig {
    /// Address to connect to (for sender) or bind to (for receiver)
    pub address: String,
    /// Port number
    pub port: u16,
    /// Auto-reconnect on connection failures
    pub auto_reconnect: bool,
    /// Delay between reconnection attempts in milliseconds
    pub reconnect_delay_ms: u64,
    /// Connection timeout in milliseconds
    pub connect_timeout_ms: u64,
}

impl Default for TransportConfig {
    fn default() -> Self {
        Self {
            address: "127.0.0.1".to_string(),
            port: 9876,
            auto_reconnect: true,
            reconnect_delay_ms: 1000,
            connect_timeout_ms: 5000,
        }
    }
}

/// TCP server that listens for client connections and sends messages
pub struct TcpServer {
    stream: Arc<Mutex<Option<TcpStream>>>,
    listener: Arc<Mutex<Option<std::net::TcpListener>>>,
    address: SocketAddr,
}

impl TcpServer {
    /// Create a new TCP server for the given address
    #[allow(clippy::missing_errors_doc)]
    pub fn new<A: ToSocketAddrs>(addr: A) -> io::Result<Self> {
        let address = addr
            .to_socket_addrs()?
            .next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Invalid address"))?;

        Ok(Self {
            stream: Arc::new(Mutex::new(None)),
            listener: Arc::new(Mutex::new(None)),
            address,
        })
    }

    /// Start listening for incoming connections
    #[allow(clippy::missing_errors_doc)]
    pub fn listen(&self) -> Result<(), TransportError> {
        let listener = std::net::TcpListener::bind(self.address)?;
        *self.listener.lock().unwrap() = Some(listener);
        log::info!("TCP server listening on {}", self.address);
        Ok(())
    }

    /// Try to accept a new client connection (non-blocking)
    fn try_accept(&self) -> Result<bool, TransportError> {
        let listener_guard = self.listener.lock().unwrap();
        let listener = listener_guard.as_ref().ok_or(TransportError::NotConnected)?;

        listener.set_nonblocking(true)?;

        match listener.accept() {
            Ok((stream, peer_addr)) => {
                stream.set_nodelay(true)?;
                log::info!("TCP server accepted connection from {peer_addr}");
                drop(listener_guard);
                *self.stream.lock().unwrap() = Some(stream);
                Ok(true)
            }
            Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => Ok(false),
            Err(e) => Err(TransportError::Io(e)),
        }
    }
}

impl Transport for TcpServer {
    fn send(&self, frame: &MessageFrame) -> Result<(), TransportError> {
        use std::io::Write;

        let mut guard = self.stream.lock().unwrap();
        let stream = guard.as_mut().ok_or(TransportError::NotConnected)?;

        let data = frame.to_bytes()?;
        let len = u32::try_from(data.len()).unwrap();

        if stream.write_all(&len.to_be_bytes()).is_err() || stream.write_all(&data).is_err() || stream.flush().is_err() {
            *guard = None;
            return Err(TransportError::ConnectionClosed);
        }

        Ok(())
    }

    fn receive(&self, _block: bool) -> Result<Option<MessageFrame>, TransportError> {
        // Servers don't receive messages, they only send
        Err(TransportError::Io(io::Error::new(
            io::ErrorKind::Unsupported,
            "Server cannot receive messages",
        )))
    }

    fn is_connected(&self) -> bool {
        self.stream.lock().unwrap().is_some()
    }

    fn connect(&self) -> Result<(), TransportError> {
        // For server: start listening and try to accept a connection
        if self.is_connected() {
            return Ok(());
        }

        if self.listener.lock().unwrap().is_none() {
            self.listen()?;
        }

        match self.try_accept() {
            Ok(true) => Ok(()),
            Ok(false) => Err(TransportError::NotConnected),
            Err(e) => Err(e),
        }
    }

    fn close(&self) {
        *self.stream.lock().unwrap() = None;
        *self.listener.lock().unwrap() = None;
    }
}

/// TCP client that connects to a server and receives messages
pub struct TcpClient {
    stream: Arc<Mutex<Option<TcpStream>>>,
    address: SocketAddr,
}

impl TcpClient {
    /// Create a new TCP client for the given address
    #[allow(clippy::missing_errors_doc)]
    pub fn new<A: ToSocketAddrs>(addr: A) -> io::Result<Self> {
        let address = addr
            .to_socket_addrs()?
            .next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Invalid address"))?;

        Ok(Self {
            stream: Arc::new(Mutex::new(None)),
            address,
        })
    }

    /// Connect to the remote server
    #[allow(clippy::missing_errors_doc)]
    pub fn connect(&self) -> Result<(), TransportError> {
        let stream = TcpStream::connect(self.address)?;
        stream.set_nodelay(true)?;
        *self.stream.lock().unwrap() = Some(stream);
        log::info!("TCP client connected to {}", self.address);
        Ok(())
    }

    /// Read a message from the stream
    fn read_message(&self, block: bool) -> Result<Option<MessageFrame>, TransportError> {
        use std::io::Read;

        let mut guard = self.stream.lock().unwrap();
        let stream = guard.as_mut().ok_or(TransportError::NotConnected)?;

        // Set blocking mode
        if !block {
            stream.set_nonblocking(true)?;
        }

        let result = (|| {
            // Try to read length prefix
            let mut len_buf = [0u8; 4];
            match stream.read_exact(&mut len_buf) {
                Ok(()) => {}
                Err(ref e) if e.kind() == io::ErrorKind::WouldBlock && !block => {
                    return Ok(None);
                }
                Err(e) => return Err(TransportError::Io(e)),
            }

            // Got the length, switch back to blocking mode for the rest
            if !block {
                stream.set_nonblocking(false)?;
            }

            let len = u32::from_be_bytes(len_buf) as usize;
            if len > 100 * 1024 * 1024 {
                return Err(TransportError::Io(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Message too large: {len} bytes"),
                )));
            }

            // Read the data
            let mut data = vec![0u8; len];
            stream.read_exact(&mut data)?;

            let frame = MessageFrame::from_bytes(&data)?;
            Ok(Some(frame))
        })();

        // Always restore blocking mode
        if !block {
            let _ = stream.set_nonblocking(false);
        }

        // If we got a connection error, clear the stream so we can reconnect
        if let Err(TransportError::Io(ref e)) = result {
            if e.kind() == io::ErrorKind::UnexpectedEof
                || e.kind() == io::ErrorKind::ConnectionReset
                || e.kind() == io::ErrorKind::ConnectionAborted
                || e.kind() == io::ErrorKind::BrokenPipe
            {
                log::warn!("TCP connection lost: {e}, clearing stream for reconnection");
                *guard = None;
                return Err(TransportError::ConnectionClosed);
            }
        }

        result
    }
}

impl Transport for TcpClient {
    fn send(&self, _frame: &MessageFrame) -> Result<(), TransportError> {
        // Clients don't send messages, they only receive
        Err(TransportError::Io(io::Error::new(
            io::ErrorKind::Unsupported,
            "Client cannot send messages",
        )))
    }

    fn receive(&self, block: bool) -> Result<Option<MessageFrame>, TransportError> {
        self.read_message(block)
    }

    fn is_connected(&self) -> bool {
        self.stream.lock().unwrap().is_some()
    }

    fn connect(&self) -> Result<(), TransportError> {
        if self.is_connected() {
            return Ok(());
        }

        match TcpStream::connect_timeout(&self.address, std::time::Duration::from_millis(5000)) {
            Ok(stream) => {
                stream.set_nodelay(true)?;
                *self.stream.lock().unwrap() = Some(stream);
                log::info!("TCP client connected to {}", self.address);
                Ok(())
            }
            Err(e) => Err(TransportError::Io(e)),
        }
    }

    fn close(&self) {
        *self.stream.lock().unwrap() = None;
    }
}
