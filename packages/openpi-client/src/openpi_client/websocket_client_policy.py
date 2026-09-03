import logging
import time
from typing import Dict, Optional, Tuple

from typing_extensions import override
import websockets.sync.client

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        jpeg_quality: Optional[int] = None,
        websocket_compression: Optional[str] = None,
    ) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        if jpeg_quality is not None and not 1 <= jpeg_quality <= 100:
            raise ValueError(f"jpeg_quality must be between 1 and 100, got {jpeg_quality}")
        if websocket_compression not in (None, "deflate"):
            raise ValueError(f"websocket_compression must be None or 'deflate', got {websocket_compression!r}")
        self._jpeg_quality = jpeg_quality
        self._websocket_compression = websocket_compression
        self._ws, self._server_metadata = self._wait_for_server()
        transport_metadata = self._server_metadata.pop("_openpi_transport", {})
        if self._jpeg_quality is not None and not transport_metadata.get("jpeg_images", False):
            self._ws.close()
            raise RuntimeError(
                "The policy server does not support JPEG image transport. Update and restart the server first."
            )

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=self._websocket_compression,
                    max_size=None,
                    additional_headers=headers,
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        pack_started = time.perf_counter()
        if self._jpeg_quality is not None:
            obs = msgpack_numpy.encode_jpeg_images(obs, self._jpeg_quality)
        data = self._packer.pack(obs)
        pack_ms = (time.perf_counter() - pack_started) * 1000.0

        send_started = time.perf_counter()
        self._ws.send(data)
        send_ms = (time.perf_counter() - send_started) * 1000.0

        recv_started = time.perf_counter()
        response = self._ws.recv()
        recv_ms = (time.perf_counter() - recv_started) * 1000.0
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")

        unpack_started = time.perf_counter()
        result = msgpack_numpy.unpackb(response)
        unpack_ms = (time.perf_counter() - unpack_started) * 1000.0
        if isinstance(result, dict):
            result["client_timing"] = {
                "pack_ms": pack_ms,
                "send_ms": send_ms,
                "recv_ms": recv_ms,
                "unpack_ms": unpack_ms,
                "request_bytes": len(data),
                "response_bytes": len(response),
            }
        return result

    @override
    def reset(self) -> None:
        pass
