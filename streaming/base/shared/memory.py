# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Improved quiet implementation of shared memory in pure python."""

import atexit
import logging
import threading
from multiprocessing import resource_tracker  # pyright: ignore
from multiprocessing.shared_memory import SharedMemory as BuiltinSharedMemory
from time import sleep
from typing import Any, Optional

from streaming.base.constant import TICK
logger = logging.getLogger(__name__)

class SharedMemory:
    """Improved quiet implementation of shared memory with better synchronization."""

    _cleanup_lock = threading.Lock()  # Class-level lock for cleanup operations

    def __init__(self,
                 name: Optional[str] = None,
                 create: Optional[bool] = None,
                 size: int = 0,
                 auto_cleanup: bool = True):
        self.created_shms = []
        self.opened_shms = []
        self.name = name
        self._cleaned_up = False
        shm = None
        
        # Save original tracker functions
        original_rtracker_reg = resource_tracker.register

        max_retries = 10
        retry_delay = TICK

        for attempt in range(max_retries):
            try:
                if create is False:
                    try:
                        # Avoid tracking shared memory resources in a process who attaches
                        resource_tracker.register = self.fix_register
                        # Attach to existing shared memory block
                        shm = BuiltinSharedMemory(name, create, size)
                        self.opened_shms.append(shm)
                        break
                    except FileNotFoundError:
                        if size > 0:
                            logger.info(f"Creating shared memory {name} with size {size}")
                            # Create new shared memory block
                            shm = BuiltinSharedMemory(name, True, size)
                            self.created_shms.append(shm)
                            break
                        else:
                            if attempt < max_retries - 1:
                                print(f"sleeping for{name} {create} {size} for attempt {attempt}")
                                sleep(retry_delay * (attempt + 1))
                                continue
                            raise FileNotFoundError(f"{name} not found and {size} is 0.")

                else:
                    try:
                        # Create new shared memory block
                        shm = BuiltinSharedMemory(name, True, size)
                        self.created_shms.append(shm)
                        break
                    except FileExistsError:
                        sleep(TICK)
                        resource_tracker.register = self.fix_register
                        # Attach to existing shared memory block
                        shm = BuiltinSharedMemory(name, False, size)
                        self.opened_shms.append(shm)
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying...")
                            sleep(retry_delay * (attempt + 1))
                            continue
                        raise

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"SharedMemory creation attempt {attempt + 1} failed: {e}")
                    sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    logger.error(f"Failed to create/attach SharedMemory after {max_retries} attempts")
                    raise
            finally:
                resource_tracker.register = original_rtracker_reg

        if shm is None:
            raise RuntimeError(f"Failed to create or attach to shared memory {name}")

        self.shm = shm

        if auto_cleanup:
            atexit.register(self.cleanup)

    @property
    def buf(self) -> memoryview:
        """Internal buffer accessor."""
        if self.shm is None:
            raise RuntimeError("SharedMemory has been cleaned up")
        return self.shm.buf

    def fix_register(self, name: str, rtype: str) -> Any:
        """Skip registering resource tracking for shared memory."""
        if rtype == 'shared_memory':
            return
        return resource_tracker._resource_tracker.register(self, name, rtype)

    def fix_unregister(self, name: str, rtype: str) -> Any:
        """Skip un-registering resource tracking for shared memory."""
        if rtype == 'shared_memory':
            return
        return resource_tracker._resource_tracker.unregister(self, name, rtype)

    def cleanup(self):
        """Clean up SharedMemory resources with proper synchronization."""
        with self._cleanup_lock:
            if self._cleaned_up:
                return
                
            # Save original unregister tracker function
            original_rtracker_unreg = resource_tracker.unregister

            try:
                # Close created SharedMemory instances
                for shm in self.created_shms:
                    try:
                        shm.close()
                        shm.unlink()
                    except Exception as e:
                        logger.warning(f"Error cleaning up created shared memory: {e}")

                # Close opened SharedMemory instances
                for shm in self.opened_shms:
                    try:
                        resource_tracker.unregister = self.fix_unregister
                        shm.close()
                    except Exception as e:
                        logger.warning(f"Error cleaning up opened shared memory: {e}")

            except Exception as e:
                logger.warning(f"Error during SharedMemory cleanup: {e}")
            finally:
                resource_tracker.unregister = original_rtracker_unreg
                self._cleaned_up = True

