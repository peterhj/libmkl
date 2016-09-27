use ffi::*;

use libc::{c_void};
use std::marker::{PhantomData};
use std::ptr::{null_mut};

#[derive(Clone, Copy)]
enum Flavor {
  F32,
  F64,
}

pub struct MklDnnAttrs<T> {
  inner:    dnnPrimitiveAttributes_t,
  flavor:   Flavor,
  _marker:  PhantomData<T>,
}

impl<T> Drop for MklDnnAttrs<T> {
  fn drop(&mut self) {
    let status = match self.flavor {
      Flavor::F32 => unsafe { dnnPrimitiveAttributesDestroy_F32(self.inner) },
      Flavor::F64 => unimplemented!(),
    };
    assert!(status.is_ok());
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
  Gemm,
  Direct,
  Fft,
}

impl MklDnnConvAlgo {
  pub fn to_ffi(&self) -> dnnAlgorithm_t {
    match *self {
      MklDnnConvAlgo::Gemm    => dnnAlgorithm_t::dnnAlgorithmConvolutionGemm,
      MklDnnConvAlgo::Direct  => dnnAlgorithm_t::dnnAlgorithmConvolutionDirect,
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
  inner:    dnnPrimitive_t,
  attrs:    MklDnnAttrs<T>,
  res:      Vec<*mut c_void>,
}

impl MklDnnConv2dFwd<f32> {
  pub fn create(cfg: MklDnnConv2dConfig) -> Result<MklDnnConv2dFwd<f32>, dnnError_t> {
    cfg.check();
    let mut inner: dnnPrimitive_t = null_mut();
    let attrs = MklDnnAttrs::create().unwrap();
    let mut offsets = vec![-(cfg.pad[0] as i32), -(cfg.pad[1] as i32)];
    let status = unsafe { dnnConvolutionCreateForwardBias_F32(
        &mut inner as *mut _,
        attrs.inner,
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
    Ok(MklDnnConv2dFwd{
      cfg:      cfg,
      inner:    inner,
      attrs:    attrs,
      res:      res,
    })
  }

  pub fn execute(&mut self, in_: *const f32, w: *const f32, b: *const f32, out: *mut f32) -> Result<(), dnnError_t> {
    self.res[dnnResourceType_t::dnnResourceSrc as usize]    = in_ as *mut _;
    self.res[dnnResourceType_t::dnnResourceFilter as usize] = w as *mut _;
    self.res[dnnResourceType_t::dnnResourceBias as usize]   = b as *mut _;
    self.res[dnnResourceType_t::dnnResourceDst as usize]    = out as *mut _;
    let status = unsafe { dnnExecute_F32(self.inner, self.res.as_mut_ptr()) };
    if status.is_err() {
      return Err(status);
    }
    Ok(())
  }
}
