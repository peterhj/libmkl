#![allow(non_camel_case_types)]

extern crate libc;

use ffi::*;

use libc::{c_void};
use std::marker::{PhantomData};
use std::ptr::{null_mut};

pub mod ffi;
pub mod prelude;

#[derive(Clone, Copy)]
enum Flavor {
  F32,
  F64,
}

pub struct MklDnnLayout<T> {
  dim:      Vec<usize>,
  stride:   Vec<usize>,
  inner:    dnnLayout_t,
  flavor:   Flavor,
  _marker:  PhantomData<fn (T)>,
}

impl<T> Drop for MklDnnLayout<T> {
  fn drop(&mut self) {
    // FIXME(20161007): leak this in case double free is causing segfault.
    /*let status = match self.flavor {
      Flavor::F32 => unsafe { dnnLayoutDestroy_F32(self.inner) },
      Flavor::F64 => unimplemented!(),
    };
    assert!(status.is_ok());*/
  }
}

impl MklDnnLayout<f32> {
  pub fn create(dim: Vec<usize>, stride: Vec<usize>) -> Result<MklDnnLayout<f32>, dnnError_t> {
    let ndim = dim.len();
    assert_eq!(ndim, stride.len());
    let mut inner: dnnLayout_t = null_mut();
    let status = unsafe { dnnLayoutCreate_F32(&mut inner as *mut _, ndim, dim.as_ptr(), stride.as_ptr()) };
    if status.is_err() {
      return Err(status);
    }
    Ok(MklDnnLayout{
      dim:      dim,
      stride:   stride,
      inner:    inner,
      flavor:   Flavor::F32,
      _marker:  PhantomData,
    })
  }
}

pub struct MklDnnAttrs<T> {
  inner:    dnnPrimitiveAttributes_t,
  flavor:   Flavor,
  _marker:  PhantomData<T>,
}

impl<T> Drop for MklDnnAttrs<T> {
  fn drop(&mut self) {
    // FIXME(20161007): leak this in case double free is causing segfault.
    /*let status = match self.flavor {
      Flavor::F32 => unsafe { dnnPrimitiveAttributesDestroy_F32(self.inner) },
      Flavor::F64 => unimplemented!(),
    };
    assert!(status.is_ok());*/
  }
}

impl MklDnnAttrs<f32> {
  pub fn create() -> Result<MklDnnAttrs<f32>, dnnError_t> {
    let mut inner: dnnPrimitiveAttributes_t = null_mut();
    let status = unsafe { dnnPrimitiveAttributesCreate_F32(&mut inner as *mut _) };
    if status.is_err() {
      return Err(status);
    }
    Ok(MklDnnAttrs{
      inner:    inner,
      flavor:   Flavor::F32,
      _marker:  PhantomData,
    })
  }
}

#[derive(Clone, Copy, Debug)]
pub enum MklDnnConvAlgo {
  Direct,
  Gemm,
  Fft,
}

impl MklDnnConvAlgo {
  pub fn to_ffi(&self) -> dnnAlgorithm_t {
    match *self {
      MklDnnConvAlgo::Direct  => dnnAlgorithm_t::dnnAlgorithmConvolutionDirect,
      MklDnnConvAlgo::Gemm    => dnnAlgorithm_t::dnnAlgorithmConvolutionGemm,
      MklDnnConvAlgo::Fft     => dnnAlgorithm_t::dnnAlgorithmConvolutionFFT,
    }
  }
}

#[derive(Clone, Debug)]
pub struct MklDnnConv2dConfig {
  pub algo:     MklDnnConvAlgo,
  pub in_dim:   Vec<usize>,
  pub out_dim:  Vec<usize>,
  pub w_dim:    Vec<usize>,
  pub stride:   Vec<usize>,
  pub pad:      Vec<usize>,
  pub bias:     bool,
}

impl MklDnnConv2dConfig {
  pub fn check(&self) {
    assert_eq!(4, self.in_dim.len());
    assert_eq!(4, self.out_dim.len());
    assert_eq!(4, self.w_dim.len());
    assert_eq!(2, self.stride.len());
    assert_eq!(2, self.pad.len());
  }
}

pub struct MklDnnConv2dFwd<T> {
  cfg:      MklDnnConv2dConfig,
  //attrs:    MklDnnAttrs<T>,
  offsets:  Vec<i32>,
  inner:    dnnPrimitive_t,
  res:      Vec<*mut c_void>,
  _marker:  PhantomData<fn (T)>,
}

impl MklDnnConv2dFwd<f32> {
  pub fn create(cfg: MklDnnConv2dConfig) -> Result<MklDnnConv2dFwd<f32>, dnnError_t> {
    cfg.check();
    let mut inner: dnnPrimitive_t = null_mut();
    //let attrs = MklDnnAttrs::create().unwrap();
    // FIXME(20161007): the dim descriptors may need to be reversed;
    // see: <https://github.com/01org/mkl-dnn/blob/master/tests/only_convolution.cpp>.
    let mut offsets = vec![-(cfg.pad[0] as i32), -(cfg.pad[1] as i32)];
    if cfg.bias {
      let status = unsafe { dnnConvolutionCreateForwardBias_F32(
          &mut inner as *mut _,
          null_mut(),
          cfg.algo.to_ffi(),
          4,
          cfg.in_dim.as_ptr(),
          cfg.out_dim.as_ptr(),
          cfg.w_dim.as_ptr(),
          cfg.stride.as_ptr(),
          offsets.as_ptr(),
          dnnBorder_t::dnnBorderZeros,
      ) };
      if status.is_err() {
        return Err(status);
      }
    } else {
      let status = unsafe { dnnConvolutionCreateForward_F32(
          &mut inner as *mut _,
          null_mut(),
          cfg.algo.to_ffi(),
          4,
          cfg.in_dim.as_ptr(),
          cfg.out_dim.as_ptr(),
          cfg.w_dim.as_ptr(),
          cfg.stride.as_ptr(),
          offsets.as_ptr(),
          dnnBorder_t::dnnBorderZeros,
      ) };
      if status.is_err() {
        return Err(status);
      }
    }
    let nres = dnnResourceType_t::dnnResourceNumber as usize;
    let mut res = Vec::with_capacity(nres);
    for _ in 0 .. nres {
      res.push(null_mut());
    }
    Ok(MklDnnConv2dFwd{
      cfg:      cfg,
      //attrs:    attrs,
      offsets:  offsets,
      inner:    inner,
      res:      res,
      _marker:  PhantomData,
    })
  }

  pub fn execute(&mut self, in_: *const f32, w: *const f32, b: Option<*const f32>, out: *mut f32) -> Result<(), dnnError_t> {
    self.res[dnnResourceType_t::dnnResourceSrc as usize]    = in_ as *mut _;
    self.res[dnnResourceType_t::dnnResourceFilter as usize] = w as *mut _;
    if self.cfg.bias {
      self.res[dnnResourceType_t::dnnResourceBias as usize] = b.unwrap() as *mut _;
    }
    self.res[dnnResourceType_t::dnnResourceDst as usize]    = out as *mut _;
    let status = unsafe { dnnExecute_F32(self.inner, self.res.as_mut_ptr()) };
    if status.is_err() {
      return Err(status);
    }
    Ok(())
  }
}

/*pub struct MklDnnConv2dFwdNoBias<T> {
  cfg:      MklDnnConv2dConfig,
  attrs:    MklDnnAttrs<T>,
  offsets:  Vec<i32>,
  inner:    dnnPrimitive_t,
  res:      Vec<*mut c_void>,
}

impl MklDnnConv2dFwdNoBias<f32> {
  pub fn create(cfg: MklDnnConv2dConfig) -> Result<MklDnnConv2dFwdNoBias<f32>, dnnError_t> {
    cfg.check();
    let mut inner: dnnPrimitive_t = null_mut();
    let attrs = MklDnnAttrs::create().unwrap();
    let mut offsets = vec![-(cfg.pad[0] as i32), -(cfg.pad[1] as i32)];
    let status = unsafe { dnnConvolutionCreateForward_F32(
        &mut inner as *mut _,
        //attrs.inner,
        null_mut(),
        cfg.algo.to_ffi(),
        4,
        cfg.in_dim.as_ptr(),
        cfg.out_dim.as_ptr(),
        cfg.w_dim.as_ptr(),
        cfg.stride.as_ptr(),
        offsets.as_ptr(),
        dnnBorder_t::dnnBorderZeros,
    ) };
    if status.is_err() {
      return Err(status);
    }
    let nres = dnnResourceType_t::dnnResourceNumber as usize;
    let mut res = Vec::with_capacity(nres);
    for _ in 0 .. nres {
      res.push(null_mut());
    }
    Ok(MklDnnConv2dFwdNoBias{
      cfg:      cfg,
      attrs:    attrs,
      offsets:  offsets,
      inner:    inner,
      res:      res,
    })
  }

  pub fn execute(&mut self, in_: *const f32, w: *const f32, out: *mut f32) -> Result<(), dnnError_t> {
    self.res[dnnResourceType_t::dnnResourceSrc as usize]    = in_ as *mut _;
    self.res[dnnResourceType_t::dnnResourceFilter as usize] = w as *mut _;
    self.res[dnnResourceType_t::dnnResourceDst as usize]    = out as *mut _;
    let status = unsafe { dnnExecute_F32(self.inner, self.res.as_mut_ptr()) };
    if status.is_err() {
      return Err(status);
    }
    Ok(())
  }
}*/

pub struct MklDnnConv2dBwdInput<T> {
  cfg:      MklDnnConv2dConfig,
  //attrs:    MklDnnAttrs<T>,
  offsets:  Vec<i32>,
  inner:    dnnPrimitive_t,
  res:      Vec<*mut c_void>,
  _marker:  PhantomData<fn (T)>,
}

impl MklDnnConv2dBwdInput<f32> {
  pub fn create(cfg: MklDnnConv2dConfig) -> Result<MklDnnConv2dBwdInput<f32>, dnnError_t> {
    cfg.check();
    let mut inner: dnnPrimitive_t = null_mut();
    //let attrs = MklDnnAttrs::create().unwrap();
    let mut offsets = vec![-(cfg.pad[0] as i32), -(cfg.pad[1] as i32)];
    //let status = unsafe { dnnGroupsConvolutionCreateBackwardData_F32(
    let status = unsafe { dnnConvolutionCreateBackwardData_F32(
        &mut inner as *mut _,
        //attrs.inner,
        null_mut(),
        cfg.algo.to_ffi(),
        //1,
        4,
        cfg.in_dim.as_ptr(),
        cfg.out_dim.as_ptr(),
        cfg.w_dim.as_ptr(),
        cfg.stride.as_ptr(),
        offsets.as_ptr(),
        dnnBorder_t::dnnBorderZeros,
    ) };
    if status.is_err() {
      return Err(status);
    }
    let nres = dnnResourceType_t::dnnResourceNumber as usize;
    let mut res = Vec::with_capacity(nres);
    for _ in 0 .. nres {
      res.push(null_mut());
    }
    Ok(MklDnnConv2dBwdInput{
      cfg:      cfg,
      //attrs:    attrs,
      offsets:  offsets,
      inner:    inner,
      res:      res,
      _marker:  PhantomData,
    })
  }

  pub fn execute(&mut self, in_buf: *const f32, w: *const f32, out_grad: *const f32, in_grad: *mut f32) -> Result<(), dnnError_t> {
    //self.res[dnnResourceType_t::dnnResourceSrc as usize]        = in_buf as *mut _;
    self.res[dnnResourceType_t::dnnResourceFilter as usize]     = w as *mut _;
    self.res[dnnResourceType_t::dnnResourceDiffDst as usize]    = out_grad as *mut _;
    self.res[dnnResourceType_t::dnnResourceDiffSrc as usize]    = in_grad as *mut _;
    let status = unsafe { dnnExecute_F32(self.inner, self.res.as_mut_ptr()) };
    if status.is_err() {
      return Err(status);
    }
    Ok(())
  }
}

pub struct MklDnnConv2dBwdKernel<T> {
  cfg:      MklDnnConv2dConfig,
  //attrs:    MklDnnAttrs<T>,
  offsets:  Vec<i32>,
  inner:    dnnPrimitive_t,
  res:      Vec<*mut c_void>,
  _marker:  PhantomData<fn (T)>,
}

impl MklDnnConv2dBwdKernel<f32> {
  pub fn create(cfg: MklDnnConv2dConfig) -> Result<MklDnnConv2dBwdKernel<f32>, dnnError_t> {
    cfg.check();
    let mut inner: dnnPrimitive_t = null_mut();
    //let attrs = MklDnnAttrs::create().unwrap();
    let mut offsets = vec![-(cfg.pad[0] as i32), -(cfg.pad[1] as i32)];
    let status = unsafe { dnnConvolutionCreateBackwardFilter_F32(
        &mut inner as *mut _,
        //attrs.inner,
        null_mut(),
        cfg.algo.to_ffi(),
        4,
        cfg.in_dim.as_ptr(),
        cfg.out_dim.as_ptr(),
        cfg.w_dim.as_ptr(),
        cfg.stride.as_ptr(),
        offsets.as_ptr(),
        dnnBorder_t::dnnBorderZeros,
    ) };
    if status.is_err() {
      return Err(status);
    }
    assert!(!inner.is_null());
    let nres = dnnResourceType_t::dnnResourceNumber as usize;
    let mut res = Vec::with_capacity(nres);
    for _ in 0 .. nres {
      res.push(null_mut());
    }
    Ok(MklDnnConv2dBwdKernel{
      cfg:      cfg,
      //attrs:    attrs,
      offsets:  offsets,
      inner:    inner,
      res:      res,
      _marker:  PhantomData,
    })
  }

  pub fn execute(&mut self, in_buf: *const f32, out_grad: *const f32, w_grad: *mut f32) -> Result<(), dnnError_t> {
    self.res[dnnResourceType_t::dnnResourceSrc as usize]        = in_buf as *mut _;
    self.res[dnnResourceType_t::dnnResourceDiffDst as usize]    = out_grad as *mut _;
    self.res[dnnResourceType_t::dnnResourceDiffFilter as usize] = w_grad as *mut _;
    let status = unsafe { dnnExecute_F32(self.inner, self.res.as_mut_ptr()) };
    if status.is_err() {
      return Err(status);
    }
    Ok(())
  }
}

pub struct MklDnnConv2dBwdBias<T> {
  cfg:      MklDnnConv2dConfig,
  //attrs:    MklDnnAttrs<T>,
  inner:    dnnPrimitive_t,
  res:      Vec<*mut c_void>,
  _marker:  PhantomData<fn (T)>,
}

impl MklDnnConv2dBwdBias<f32> {
  pub fn create(cfg: MklDnnConv2dConfig) -> Result<MklDnnConv2dBwdBias<f32>, dnnError_t> {
    cfg.check();
    let mut inner: dnnPrimitive_t = null_mut();
    //let attrs = MklDnnAttrs::create().unwrap();
    let status = unsafe { dnnConvolutionCreateBackwardBias_F32(
        &mut inner as *mut _,
        //attrs.inner,
        null_mut(),
        cfg.algo.to_ffi(),
        4,
        cfg.out_dim.as_ptr(),
    ) };
    if status.is_err() {
      return Err(status);
    }
    let nres = dnnResourceType_t::dnnResourceNumber as usize;
    let mut res = Vec::with_capacity(nres);
    for _ in 0 .. nres {
      res.push(null_mut());
    }
    Ok(MklDnnConv2dBwdBias{
      cfg:      cfg,
      inner:    inner,
      //attrs:    attrs,
      res:      res,
      _marker:  PhantomData,
    })
  }

  pub fn execute(&mut self, out_grad: *const f32, b_grad: *mut f32) -> Result<(), dnnError_t> {
    self.res[dnnResourceType_t::dnnResourceDiffDst as usize]    = out_grad as *mut _;
    self.res[dnnResourceType_t::dnnResourceDiffBias as usize]   = b_grad as *mut _;
    let status = unsafe { dnnExecute_F32(self.inner, self.res.as_mut_ptr()) };
    if status.is_err() {
      return Err(status);
    }
    Ok(())
  }
}

#[derive(Clone, Copy, Debug)]
pub enum MklDnnPoolAlgo {
  Max,
  Min,
  Average,
}

impl MklDnnPoolAlgo {
  pub fn to_ffi(&self) -> dnnAlgorithm_t {
    match *self {
      MklDnnPoolAlgo::Max       => dnnAlgorithm_t::dnnAlgorithmPoolingMax,
      MklDnnPoolAlgo::Min       => dnnAlgorithm_t::dnnAlgorithmPoolingMin,
      MklDnnPoolAlgo::Average   => dnnAlgorithm_t::dnnAlgorithmPoolingAvg,
    }
  }
}

#[derive(Clone, Debug)]
pub struct MklDnnPool2dConfig {
  pub in_dim:   Vec<usize>,
  pub pool_dim: Vec<usize>,
  pub stride:   Vec<usize>,
  pub pad:      Vec<usize>,
  pub algo:     MklDnnPoolAlgo,
}

pub struct MklDnnPool2dFwd<T> {
  cfg:      MklDnnPool2dConfig,
  offsets:  Vec<i32>,
  inner:    dnnPrimitive_t,
  res:      Vec<*mut c_void>,
  _marker:  PhantomData<fn (T)>,
}

impl MklDnnPool2dFwd<f32> {
  pub fn create(cfg: MklDnnPool2dConfig) -> Result<MklDnnPool2dFwd<f32>, dnnError_t> {
    let mut inner: dnnPrimitive_t = null_mut();
    let in_strides = vec![
        1,
        cfg.in_dim[0],
        cfg.in_dim[0] * cfg.in_dim[1],
        cfg.in_dim[0] * cfg.in_dim[1] * cfg.in_dim[2],
    ];
    let mut layout = MklDnnLayout::create(cfg.in_dim.clone(), in_strides).unwrap();
    let mut offsets = vec![-(cfg.pad[0] as i32), -(cfg.pad[1] as i32)];
    let status = unsafe { dnnPoolingCreateForward_F32(
        &mut inner as *mut _,
        null_mut(),
        cfg.algo.to_ffi(),
        layout.inner,
        cfg.pool_dim.as_ptr(),
        cfg.stride.as_ptr(),
        offsets.as_ptr(),
        dnnBorder_t::dnnBorderZeros,
    ) };
    if status.is_err() {
      return Err(status);
    }
    assert!(!inner.is_null());
    let nres = dnnResourceType_t::dnnResourceNumber as usize;
    let mut res = Vec::with_capacity(nres);
    for _ in 0 .. nres {
      res.push(null_mut());
    }
    Ok(MklDnnPool2dFwd{
      cfg:      cfg,
      offsets:  offsets,
      inner:    inner,
      res:      res,
      _marker:  PhantomData,
    })
  }
}

pub struct MklDnnPool2dBwd<T> {
  cfg:      MklDnnPool2dConfig,
  offsets:  Vec<i32>,
  inner:    dnnPrimitive_t,
  res:      Vec<*mut c_void>,
  _marker:  PhantomData<fn (T)>,
}

impl MklDnnPool2dBwd<f32> {
  pub fn create(cfg: MklDnnPool2dConfig) -> Result<MklDnnPool2dBwd<f32>, dnnError_t> {
    let mut inner: dnnPrimitive_t = null_mut();
    let in_strides = vec![
        1,
        cfg.in_dim[0],
        cfg.in_dim[0] * cfg.in_dim[1],
        cfg.in_dim[0] * cfg.in_dim[1] * cfg.in_dim[2],
    ];
    let mut layout = MklDnnLayout::create(cfg.in_dim.clone(), in_strides).unwrap();
    let mut offsets = vec![-(cfg.pad[0] as i32), -(cfg.pad[1] as i32)];
    let status = unsafe { dnnPoolingCreateBackward_F32(
        &mut inner as *mut _,
        null_mut(),
        cfg.algo.to_ffi(),
        layout.inner,
        cfg.pool_dim.as_ptr(),
        cfg.stride.as_ptr(),
        offsets.as_ptr(),
        dnnBorder_t::dnnBorderZeros,
    ) };
    if status.is_err() {
      return Err(status);
    }
    assert!(!inner.is_null());
    let nres = dnnResourceType_t::dnnResourceNumber as usize;
    let mut res = Vec::with_capacity(nres);
    for _ in 0 .. nres {
      res.push(null_mut());
    }
    Ok(MklDnnPool2dBwd{
      cfg:      cfg,
      offsets:  offsets,
      inner:    inner,
      res:      res,
      _marker:  PhantomData,
    })
  }

  pub fn execute(&mut self, in_buf: *const f32, out_grad: *const f32) -> Result<(), dnnError_t> {
    self.res[dnnResourceType_t::dnnResourceSrc as usize]        = in_buf as *mut _;
    self.res[dnnResourceType_t::dnnResourceDiffDst as usize]    = out_grad as *mut _;
    let status = unsafe { dnnExecute_F32(self.inner, self.res.as_mut_ptr()) };
    if status.is_err() {
      return Err(status);
    }
    Ok(())
  }
}
