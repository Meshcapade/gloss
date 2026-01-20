#![allow(non_snake_case)]
#![allow(non_camel_case_types)]

use cust_raw::CUdeviceptr;
use std::ffi::c_void;
// use cust::error::CudaResultExt;
use crate::cuda_ext::CudaResultExt;
use crate::AllocSize;
use log::error;

pub struct CudaSharedMemory {
    pub device_ptr: CUdeviceptr,
    pub shared_handle: usize,
    pub cuda_alloc_size: usize,
    pub vulkan_pitch_alignment: usize,
    pub alloc_size: AllocSize, //size used in the constructor used to create the CudaSharedMemory. Useful for checking if we need to recreate this memory or if we can recycly
}
impl Drop for CudaSharedMemory {
    fn drop(&mut self) {
        unsafe {
            // Unmap the memory
            let unmap_result = cust_raw::cuMemUnmap(self.device_ptr, self.cuda_alloc_size);
            if unmap_result != cust_raw::CUresult::CUDA_SUCCESS {
                error!("Failed to unmap memory: {unmap_result:?}");
            }

            // Free the reserved address range
            let address_free_result = cust_raw::cuMemAddressFree(self.device_ptr, self.cuda_alloc_size);
            if address_free_result != cust_raw::CUresult::CUDA_SUCCESS {
                error!("Failed to free virtual address space: {address_free_result:?}");
            }

            // let fd = self.shared_handle as std::os::unix::prelude::RawFd;
            let rc = libc::close(i32::try_from(self.shared_handle).unwrap());
            if rc != 0 {
                // handle close(2) error if you care (unlikely)
                eprintln!("warning: close(fd) failed: {}", std::io::Error::last_os_error());
            }
        }
    }
}

impl CudaSharedMemory {
    //more info in https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/simpleVulkanMMAP
    /// Allocates CUDA shared memory exported as a POSIX file descriptor for interop with Vulkan.
    pub fn new(alloc_size: AllocSize) -> Self {
        let size = alloc_size.full_size_padded();
        unsafe {
            // Get current CUDA device
            //TODO we should read the device from CUDA_VISIBLE_DEVICES if it exists
            let mut dev: cust_raw::CUdevice = 0;
            cust_raw::cuInit(0).to_result().expect("Failed to initialize CUDA");
            cust_raw::cuDeviceGet(&raw mut dev, 0).to_result().expect("Failed to get CUDA device");

            // Check for existing context
            let mut existing_ctx: cust_raw::CUcontext = std::ptr::null_mut();
            cust_raw::cuCtxGetCurrent(&raw mut existing_ctx)
                .to_result()
                .expect("Failed to get current CUDA context");

            // We'll only retain & set primary if no context exists
            let mut primary_ctx: cust_raw::CUcontext = std::ptr::null_mut();
            #[allow(unused_assignments)]
            let _created_ctx = if existing_ctx.is_null() {
                cust_raw::cuDevicePrimaryCtxRetain(&raw mut primary_ctx, dev)
                    .to_result()
                    .expect("Failed to retain primary CUDA context");
                cust_raw::cuCtxSetCurrent(primary_ctx)
                    .to_result()
                    .expect("Failed to set current CUDA context");
                true
            } else {
                primary_ctx = existing_ctx;
                false
            };

            // Set up allocation properties
            let share_type = cust_raw::CUmemAllocationHandleType_enum::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
            let location = cust_raw::CUmemLocation {
                type_: cust_raw::CUmemLocationType_enum::CU_MEM_LOCATION_TYPE_DEVICE,
                id: dev,
            };
            let prop = cust_raw::CUmemAllocationProp {
                type_: cust_raw::CUmemAllocationType_enum::CU_MEM_ALLOCATION_TYPE_PINNED,
                requestedHandleTypes: share_type,
                location,
                win32HandleMetaData: std::ptr::null_mut(),
                allocFlags: cust_raw::CUmemAllocationProp_st__bindgen_ty_1::default(),
            };

            // Determine granularity & align size
            let mut granularity = 0;
            cust_raw::cuMemGetAllocationGranularity(
                &raw mut granularity,
                &raw const prop,
                cust_raw::CUmemAllocationGranularity_flags_enum::CU_MEM_ALLOC_GRANULARITY_MINIMUM,
            )
            .to_result()
            .expect("Failed to get allocation granularity");
            let aligned = (size + granularity - 1) & !(granularity - 1);

            // Reserve, create and export
            let mut device_ptr: cust_raw::CUdeviceptr = 0;
            cust_raw::cuMemAddressReserve(&raw mut device_ptr, aligned, granularity, 0, 0)
                .to_result()
                .expect("Failed to reserve device address");

            let mut alloc_handle: cust_raw::CUmemGenericAllocationHandle = 0;
            cust_raw::cuMemCreate(&raw mut alloc_handle, aligned, &raw const prop, 0)
                .to_result()
                .expect("Failed to create CUDA memory");

            let mut shared_handle: usize = 0;
            cust_raw::cuMemExportToShareableHandle((&raw mut shared_handle).cast::<c_void>(), alloc_handle, share_type, 0)
                .to_result()
                .expect("Failed to export CUDA memory to shareable handle");

            // Map & release
            cust_raw::cuMemMap(device_ptr, aligned, 0, alloc_handle, 0)
                .to_result()
                .expect("Failed to map CUDA memory");
            cust_raw::cuMemRelease(alloc_handle).to_result().expect("Failed to release CUDA memory");

            // Set access
            let desc = cust_raw::CUmemAccessDesc_st {
                location,
                flags: cust_raw::CUmemAccess_flags_enum::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
            };
            cust_raw::cuMemSetAccess(device_ptr, aligned, &raw const desc, 1)
                .to_result()
                .expect("Failed to set CUDA memory access");

            Self {
                device_ptr,
                shared_handle,
                cuda_alloc_size: aligned,
                vulkan_pitch_alignment: 1,
                alloc_size,
            }
        }
    }
}

fn align(a: usize, b: usize) -> usize {
    a.div_ceil(b) * b
}

/// Copies a 2D region from `src` to `dst` on the CUDA device.
///
/// # Errors
///
/// Returns an `Err` if any underlying CUDA operation fails; CUDA errors from `cust_raw`
/// are propagated and returned as a `Box<dyn std::error::Error>`.
pub fn cuda_2d_copy_on_device(
    size: AllocSize,
    dst: CUdeviceptr,
    src: CUdeviceptr,
    dst_alignment: usize,
    src_alignment: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let height = size.height;
    let desc = cust_raw::CUDA_MEMCPY2D_st {
        Height: height,
        WidthInBytes: size.stride,

        dstPitch: align(size.stride_padded(), dst_alignment),
        dstDevice: dst,
        srcPitch: align(size.stride, src_alignment),
        srcDevice: src,

        dstArray: std::ptr::null_mut(),
        dstHost: std::ptr::null_mut(),
        dstMemoryType: cust_raw::CUmemorytype::CU_MEMORYTYPE_DEVICE,
        dstXInBytes: 0,
        dstY: 0,
        srcArray: std::ptr::null_mut(),
        srcHost: std::ptr::null_mut(),
        srcMemoryType: cust_raw::CUmemorytype::CU_MEMORYTYPE_DEVICE,
        srcXInBytes: 0,
        srcY: 0,
    };
    unsafe { cust_raw::cuMemcpy2D_v2(&raw const desc).to_result() }
}

pub fn cuda_synchronize() {
    unsafe { cust_raw::cuCtxSynchronize() };
}
