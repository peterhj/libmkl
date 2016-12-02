#![allow(non_camel_case_types)]

extern crate libc;

use ffi::*;

use libc::{c_void};
use std::marker::{PhantomData};
use std::ptr::{copy, null_mut};

pub mod ffi;
pub mod prelude;

#[derive(Clone, Copy)]
enum Flavor {
  F32,
  F64,
}

pub struct MklDnnLayout<T> {
  dim:      Vec<usize>,
  count:    usize,
  stride:   Vec<usize>,
  inner:    dnnLayout_t,
  mem_sz:   usize,
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
  pub fn create(dim: Vec<usize>) -> Result<MklDnnLayout<f32>, dnnError_t> {
    let ndim = dim.len();
    let mut count = 1;
    let mut stride = Vec::with_capacity(ndim);
    stride.push(1);
    for i in 0 .. ndim-1 {
      let d = dim[i];
      let s = stride[i];
      count *= d;
      stride.push(s * d);
    }
    count *= dim[ndim-1];
    assert_eq!(ndim, stride.len());
    let mut inner: dnnLayout_t = null_mut();
    let status = unsafe { dnnLayoutCreate_F32(&mut inner as *mut _, ndim, dim.as_ptr(), stride.as_ptr()) };
    if status.is_err() {
      return Err(status);
    }
    let mem_sz = unsafe { dnnLayoutGetMemorySize_F32(inner) };
    assert!(count <= mem_sz);
    Ok(MklDnnLayout{
      dim:      dim,
      count:    count,
      stride:   stride,
      inner:    inner,
      mem_sz:   mem_sz,
      flavor:   Flavor::F32,
      _marker:  PhantomData,
    })
  }

  pub fn create_from_primitive(primitive: dnnPrimitive_t, res_ty: dnnResourceType_t) -> Result<MklDnnLayout<f32>, dnnError_t> {
    let mut inner: dnnLayout_t = null_mut();
    let status = unsafe { dnnLayoutCreateFromPrimitive_F32(&mut inner as *mut _, primitive, res_ty) };
    if status.is_err() {
      return Err(status);
    }
    let mem_sz = unsafe { dnnLayoutGetMemorySize_F32(inner) };
    Ok(MklDnnLayout{
      dim:      vec![],
      count:    0,
      stride:   vec![],
      inner:    inner,
      mem_sz:   mem_sz,
      flavor:   Flavor::F32,
      _marker:  PhantomData,
    })
  }
}

pub struct MklDnnBuffer<T> {
  ptr:      *mut T,
  layout:   MklDnnLayout<T>,
}

impl MklDnnBuffer<f32> {
  pub fn create(layout: MklDnnLayout<f32>) -> Result<MklDnnBuffer<f32>, dnnError_t> {
    let mut ptr = null_mut();
    let status = unsafe { dnnAllocateBuffer_F32(&mut ptr as *mut *mut c_void, layout.inner) };
    if status.is_err() {
      return Err(status);
    }
    Ok(MklDnnBuffer{
      ptr:      ptr as *mut f32,
      layout:   layout,
    })
  }
}

impl<T> MklDnnBuffer<T> {
  pub fn as_ptr(&self) -> *const T {
    self.ptr
  }

  pub fn as_mut_ptr(&self) -> *mut T {
    self.ptr
  }

  pub fn copy_from(&mut self, src: &[T]) {
    assert_eq!(self.layout.count, src.len());
    unsafe { copy(src.as_ptr(), self.as_mut_ptr(), self.layout.count) };
  }

  pub fn convert_from(&mut self, src_layout: &MklDnnLayout<T>, src: *const T) {
    unimplemented!();
  }

  pub fn copy_to(&self, dst: &mut [T]) {
    assert_eq!(self.layout.count, dst.len());
    unsafe { copy(self.as_ptr(), dst.as_mut_ptr(), self.layout.count) };
  }

  pub fn convert_to(&self, dst_layout: &MklDnnLayout<T>, dst: *mut T) {
    unimplemented!();
  }
}

pub struct MklDnnConversion<T> {
  inner:        dnnPrimitive_t,
  //from_layout:  MklDnnLayout<T>,
  //to_layout:    MklDnnLayout<T>,
  _marker:      PhantomData<fn (T)>,
}

impl MklDnnConversion<f32> {
  pub fn create(from_layout: &MklDnnLayout<f32>, to_layout: &MklDnnLayout<f32>) -> Result<Self, dnnError_t> {
    let mut inner = null_mut();
    let status = unsafe { dnnConversionCreate_F32(&mut inner as *mut _, from_layout.inner, to_layout.inner) };
    if status.is_err() {
      return Err(status);
    }
    Ok(MklDnnConversion{
      inner:        inner,
      //from_layout:  from_layout,
      //to_layout:    to_layout,
      _marker:      PhantomData,
    })
  }

  pub fn convert(&mut self, src: *const f32, dst: *mut f32) -> Result<(), dnnError_t> {
    let status = unsafe { dnnConversionExecute_F32(self.inner, src as *const _, dst as *mut _) };
    if status.is_err() {
      return Err(status);
    }
    Ok(())
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
  b_dim:    Vec<usize>,
  offsets:  Vec<i32>,
  inner:    dnnPrimitive_t,
  res:      Vec<*mut c_void>,
  src_layout:   MklDnnLayout<T>,
  w_layout:     MklDnnLayout<T>,
  b_layout:     MklDnnLayout<T>,
  dst_layout:   MklDnnLayout<T>,
  src_buf:      MklDnnBuffer<T>,
  w_buf:        MklDnnBuffer<T>,
  b_buf:        Option<MklDnnBuffer<T>>,
  dst_buf:      MklDnnBuffer<T>,
  load_src:     MklDnnConversion<T>,
  load_w:       MklDnnConversion<T>,
  load_b:       Option<MklDnnConversion<T>>,
  load_dst:     MklDnnConversion<T>,
  store_src:    MklDnnConversion<T>,
  store_w:      MklDnnConversion<T>,
  store_b:      Option<MklDnnConversion<T>>,
  store_dst:    MklDnnConversion<T>,
  _marker:  PhantomData<fn (T)>,
}

impl MklDnnConv2dFwd<f32> {
  pub fn create(cfg: MklDnnConv2dConfig) -> Result<MklDnnConv2dFwd<f32>, dnnError_t> {
    cfg.check();
    let mut inner: dnnPrimitive_t = null_mut();
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
    let b_dim = vec![cfg.w_dim[3]];
    let src_layout = MklDnnLayout::create(cfg.in_dim.clone()).unwrap();
    let w_layout = MklDnnLayout::create(cfg.w_dim.clone()).unwrap();
    //println!("DEBUG: creating bias layout");
    let b_layout = MklDnnLayout::create(b_dim.clone()).unwrap();
    //println!("DEBUG: created bias layout");
    let dst_layout = MklDnnLayout::create(cfg.out_dim.clone()).unwrap();
    let src_buf = MklDnnBuffer::create(MklDnnLayout::create_from_primitive(inner, dnnResourceType_t::dnnResourceSrc).unwrap()).unwrap();
    let w_buf = MklDnnBuffer::create(MklDnnLayout::create_from_primitive(inner, dnnResourceType_t::dnnResourceFilter).unwrap()).unwrap();
    //println!("DEBUG: creating bias buf");
    let b_buf = if cfg.bias {
      Some(MklDnnBuffer::create(MklDnnLayout::create_from_primitive(inner, dnnResourceType_t::dnnResourceBias).unwrap()).unwrap())
    } else {
      None
    };
    //println!("DEBUG: created bias buf");
    let dst_buf = MklDnnBuffer::create(MklDnnLayout::create_from_primitive(inner, dnnResourceType_t::dnnResourceDst).unwrap()).unwrap();
    let load_src = MklDnnConversion::create(&src_layout, &src_buf.layout).unwrap();
    let load_w = MklDnnConversion::create(&w_layout, &w_buf.layout).unwrap();
    //println!("DEBUG: creating bias load");
    let load_b = if cfg.bias {
      Some(MklDnnConversion::create(&b_layout, &b_buf.as_ref().unwrap().layout).unwrap())
    } else {
      None
    };
    //println!("DEBUG: created bias load");
    let load_dst = MklDnnConversion::create(&dst_layout, &dst_buf.layout).unwrap();
    let store_src = MklDnnConversion::create(&src_buf.layout, &src_layout).unwrap();
    let store_w = MklDnnConversion::create(&w_buf.layout, &w_layout).unwrap();
    //println!("DEBUG: creating bias store");
    let store_b = if cfg.bias {
      Some(MklDnnConversion::create(&b_buf.as_ref().unwrap().layout, &b_layout).unwrap())
    } else {
      None
    };
    //println!("DEBUG: created bias store");
    let store_dst = MklDnnConversion::create(&dst_buf.layout, &dst_layout).unwrap();
    Ok(MklDnnConv2dFwd{
      cfg:      cfg,
      //attrs:    attrs,
      b_dim:    b_dim,
      offsets:  offsets,
      inner:    inner,
      res:      res,
      src_layout:   src_layout,
      w_layout:     w_layout,
      b_layout:     b_layout,
      dst_layout:   dst_layout,
      src_buf:      src_buf,
      w_buf:        w_buf,
      b_buf:        b_buf,
      dst_buf:      dst_buf,
      load_src:     load_src,
      load_w:       load_w,
      load_b:       load_b,
      load_dst:     load_dst,
      store_src:    store_src,
      store_w:      store_w,
      store_b:      store_b,
      store_dst:    store_dst,
      _marker:  PhantomData,
    })
  }

  pub fn execute(&mut self, in_: *const f32, w: *const f32, b: Option<*const f32>, out: *mut f32) -> Result<(), dnnError_t> {
    /*self.res[dnnResourceType_t::dnnResourceSrc as usize]    = in_ as *mut _;
    self.res[dnnResourceType_t::dnnResourceFilter as usize] = w as *mut _;
    if self.cfg.bias {
      self.res[dnnResourceType_t::dnnResourceBias as usize] = b.unwrap() as *mut _;
    }
    self.res[dnnResourceType_t::dnnResourceDst as usize]    = out as *mut _;*/

    self.res[dnnResourceType_t::dnnResourceSrc as usize]    = self.src_buf.as_mut_ptr() as *mut _;
    self.res[dnnResourceType_t::dnnResourceFilter as usize] = self.w_buf.as_mut_ptr() as *mut _;
    if self.cfg.bias {
      self.res[dnnResourceType_t::dnnResourceBias as usize] = self.b_buf.as_ref().unwrap().as_mut_ptr() as *mut _;
    } else {
      self.res[dnnResourceType_t::dnnResourceBias as usize] = null_mut();
    }
    self.res[dnnResourceType_t::dnnResourceDst as usize]    = self.dst_buf.as_mut_ptr() as *mut _;

    self.load_src.convert(in_, self.src_buf.as_mut_ptr()).unwrap();
    self.load_w.convert(w, self.w_buf.as_mut_ptr()).unwrap();
    if self.cfg.bias {
      self.load_b.as_mut().unwrap().convert(b.unwrap(), self.b_buf.as_ref().unwrap().as_mut_ptr()).unwrap();
    }

    let status = unsafe { dnnExecute_F32(self.inner, self.res.as_mut_ptr()) };
    if status.is_err() {
      return Err(status);
    }

    self.store_dst.convert(self.dst_buf.as_ptr(), out).unwrap();

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
  src_layout:   MklDnnLayout<T>,
  w_layout:     MklDnnLayout<T>,
  dst_layout:   MklDnnLayout<T>,
  src_buf:      MklDnnBuffer<T>,
  w_buf:        MklDnnBuffer<T>,
  dst_buf:      MklDnnBuffer<T>,
  load_src:     MklDnnConversion<T>,
  load_w:       MklDnnConversion<T>,
  load_dst:     MklDnnConversion<T>,
  store_src:    MklDnnConversion<T>,
  store_w:      MklDnnConversion<T>,
  store_dst:    MklDnnConversion<T>,
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
    let src_layout = MklDnnLayout::create(cfg.in_dim.clone()).unwrap();
    let w_layout = MklDnnLayout::create(cfg.w_dim.clone()).unwrap();
    let dst_layout = MklDnnLayout::create(cfg.out_dim.clone()).unwrap();
    let src_buf = MklDnnBuffer::create(MklDnnLayout::create_from_primitive(inner, dnnResourceType_t::dnnResourceDiffSrc).unwrap()).unwrap();
    let w_buf = MklDnnBuffer::create(MklDnnLayout::create_from_primitive(inner, dnnResourceType_t::dnnResourceFilter).unwrap()).unwrap();
    let dst_buf = MklDnnBuffer::create(MklDnnLayout::create_from_primitive(inner, dnnResourceType_t::dnnResourceDiffDst).unwrap()).unwrap();
    let load_src = MklDnnConversion::create(&src_layout, &src_buf.layout).unwrap();
    let load_w = MklDnnConversion::create(&w_layout, &w_buf.layout).unwrap();
    let load_dst = MklDnnConversion::create(&dst_layout, &dst_buf.layout).unwrap();
    let store_src = MklDnnConversion::create(&src_buf.layout, &src_layout).unwrap();
    let store_w = MklDnnConversion::create(&w_buf.layout, &w_layout).unwrap();
    let store_dst = MklDnnConversion::create(&dst_buf.layout, &dst_layout).unwrap();
    Ok(MklDnnConv2dBwdInput{
      cfg:      cfg,
      //attrs:    attrs,
      offsets:  offsets,
      inner:    inner,
      res:      res,
      src_layout:   src_layout,
      w_layout:     w_layout,
      dst_layout:   dst_layout,
      src_buf:      src_buf,
      w_buf:        w_buf,
      dst_buf:      dst_buf,
      load_src:     load_src,
      load_w:       load_w,
      load_dst:     load_dst,
      store_src:    store_src,
      store_w:      store_w,
      store_dst:    store_dst,
      _marker:  PhantomData,
    })
  }

  pub fn execute(&mut self, w: *const f32, out_grad: *const f32, in_grad: *mut f32) -> Result<(), dnnError_t> {
    /*self.res[dnnResourceType_t::dnnResourceFilter as usize]     = w as *mut _;
    self.res[dnnResourceType_t::dnnResourceDiffDst as usize]    = out_grad as *mut _;
    self.res[dnnResourceType_t::dnnResourceDiffSrc as usize]    = in_grad as *mut _;*/

    self.res[dnnResourceType_t::dnnResourceFilter as usize]     = self.w_buf.as_mut_ptr() as *mut _;
    self.res[dnnResourceType_t::dnnResourceDiffDst as usize]    = self.dst_buf.as_mut_ptr() as *mut _;
    self.res[dnnResourceType_t::dnnResourceDiffSrc as usize]    = self.src_buf.as_mut_ptr() as *mut _;

    self.load_w.convert(w, self.w_buf.as_mut_ptr()).unwrap();
    self.load_dst.convert(out_grad, self.dst_buf.as_mut_ptr()).unwrap();

    let status = unsafe { dnnExecute_F32(self.inner, self.res.as_mut_ptr()) };
    if status.is_err() {
      return Err(status);
    }

    self.store_src.convert(self.src_buf.as_ptr(), in_grad).unwrap();

    Ok(())
  }
}

pub struct MklDnnConv2dBwdKernel<T> {
  cfg:      MklDnnConv2dConfig,
  //attrs:    MklDnnAttrs<T>,
  offsets:  Vec<i32>,
  inner:    dnnPrimitive_t,
  res:      Vec<*mut c_void>,
  src_layout:   MklDnnLayout<T>,
  w_layout:     MklDnnLayout<T>,
  dst_layout:   MklDnnLayout<T>,
  src_buf:      MklDnnBuffer<T>,
  w_buf:        MklDnnBuffer<T>,
  dst_buf:      MklDnnBuffer<T>,
  load_src:     MklDnnConversion<T>,
  load_w:       MklDnnConversion<T>,
  load_dst:     MklDnnConversion<T>,
  store_src:    MklDnnConversion<T>,
  store_w:      MklDnnConversion<T>,
  store_dst:    MklDnnConversion<T>,
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
    let src_layout = MklDnnLayout::create(cfg.in_dim.clone()).unwrap();
    let w_layout = MklDnnLayout::create(cfg.w_dim.clone()).unwrap();
    let dst_layout = MklDnnLayout::create(cfg.out_dim.clone()).unwrap();
    let src_buf = MklDnnBuffer::create(MklDnnLayout::create_from_primitive(inner, dnnResourceType_t::dnnResourceSrc).unwrap()).unwrap();
    let w_buf = MklDnnBuffer::create(MklDnnLayout::create_from_primitive(inner, dnnResourceType_t::dnnResourceDiffFilter).unwrap()).unwrap();
    let dst_buf = MklDnnBuffer::create(MklDnnLayout::create_from_primitive(inner, dnnResourceType_t::dnnResourceDiffDst).unwrap()).unwrap();
    let load_src = MklDnnConversion::create(&src_layout, &src_buf.layout).unwrap();
    let load_w = MklDnnConversion::create(&w_layout, &w_buf.layout).unwrap();
    let load_dst = MklDnnConversion::create(&dst_layout, &dst_buf.layout).unwrap();
    let store_src = MklDnnConversion::create(&src_buf.layout, &src_layout).unwrap();
    let store_w = MklDnnConversion::create(&w_buf.layout, &w_layout).unwrap();
    let store_dst = MklDnnConversion::create(&dst_buf.layout, &dst_layout).unwrap();
    Ok(MklDnnConv2dBwdKernel{
      cfg:      cfg,
      //attrs:    attrs,
      offsets:  offsets,
      inner:    inner,
      res:      res,
      src_layout:   src_layout,
      w_layout:     w_layout,
      dst_layout:   dst_layout,
      src_buf:      src_buf,
      w_buf:        w_buf,
      dst_buf:      dst_buf,
      load_src:     load_src,
      load_w:       load_w,
      load_dst:     load_dst,
      store_src:    store_src,
      store_w:      store_w,
      store_dst:    store_dst,
      _marker:  PhantomData,
    })
  }

  pub fn execute(&mut self, in_buf: *const f32, out_grad: *const f32, w_grad: *mut f32) -> Result<(), dnnError_t> {
    /*self.res[dnnResourceType_t::dnnResourceSrc as usize]        = in_buf as *mut _;
    self.res[dnnResourceType_t::dnnResourceDiffDst as usize]    = out_grad as *mut _;
    self.res[dnnResourceType_t::dnnResourceDiffFilter as usize] = w_grad as *mut _;*/

    self.res[dnnResourceType_t::dnnResourceSrc as usize]        = self.src_buf.as_mut_ptr() as *mut _;
    self.res[dnnResourceType_t::dnnResourceDiffDst as usize]    = self.dst_buf.as_mut_ptr() as *mut _;
    self.res[dnnResourceType_t::dnnResourceDiffFilter as usize] = self.w_buf.as_mut_ptr() as *mut _;

    self.load_src.convert(in_buf, self.src_buf.as_mut_ptr()).unwrap();
    self.load_dst.convert(out_grad, self.dst_buf.as_mut_ptr()).unwrap();

    let status = unsafe { dnnExecute_F32(self.inner, self.res.as_mut_ptr()) };
    if status.is_err() {
      return Err(status);
    }

    self.store_w.convert(self.w_buf.as_ptr(), w_grad).unwrap();

    Ok(())
  }
}

pub struct MklDnnConv2dBwdBias<T> {
  cfg:      MklDnnConv2dConfig,
  b_dim:    Vec<usize>,
  //attrs:    MklDnnAttrs<T>,
  inner:    dnnPrimitive_t,
  res:      Vec<*mut c_void>,
  b_layout:     MklDnnLayout<T>,
  dst_layout:   MklDnnLayout<T>,
  b_buf:        MklDnnBuffer<T>,
  dst_buf:      MklDnnBuffer<T>,
  load_dst:     MklDnnConversion<T>,
  store_b:      MklDnnConversion<T>,
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
    let b_dim = vec![cfg.w_dim[3]];
    let b_layout = MklDnnLayout::create(b_dim.clone()).unwrap();
    let dst_layout = MklDnnLayout::create(cfg.out_dim.clone()).unwrap();
    let b_buf = MklDnnBuffer::create(MklDnnLayout::create_from_primitive(inner, dnnResourceType_t::dnnResourceDiffBias).unwrap()).unwrap();
    let dst_buf = MklDnnBuffer::create(MklDnnLayout::create_from_primitive(inner, dnnResourceType_t::dnnResourceDiffDst).unwrap()).unwrap();
    let load_dst = MklDnnConversion::create(&dst_layout, &dst_buf.layout).unwrap();
    let store_b = MklDnnConversion::create(&b_buf.layout, &b_layout).unwrap();
    Ok(MklDnnConv2dBwdBias{
      cfg:      cfg,
      b_dim:    b_dim,
      inner:    inner,
      //attrs:    attrs,
      res:      res,
      b_layout:     b_layout,
      dst_layout:   dst_layout,
      b_buf:        b_buf,
      dst_buf:      dst_buf,
      load_dst:     load_dst,
      store_b:      store_b,
      _marker:  PhantomData,
    })
  }

  pub fn execute(&mut self, out_grad: *const f32, b_grad: *mut f32) -> Result<(), dnnError_t> {
    /*self.res[dnnResourceType_t::dnnResourceDiffDst as usize]    = out_grad as *mut _;
    self.res[dnnResourceType_t::dnnResourceDiffBias as usize]   = b_grad as *mut _;*/

    self.res[dnnResourceType_t::dnnResourceDiffDst as usize]    = self.dst_buf.as_mut_ptr() as *mut _;
    self.res[dnnResourceType_t::dnnResourceDiffBias as usize]   = self.b_buf.as_mut_ptr() as *mut _;

    self.load_dst.convert(out_grad, self.dst_buf.as_mut_ptr()).unwrap();

    let status = unsafe { dnnExecute_F32(self.inner, self.res.as_mut_ptr()) };
    if status.is_err() {
      return Err(status);
    }

    self.store_b.convert(self.b_buf.as_ptr(), b_grad).unwrap();

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
  pub out_dim:   Vec<usize>,
  pub pool_dim: Vec<usize>,
  pub stride:   Vec<usize>,
  pub pad:      Vec<usize>,
  pub algo:     MklDnnPoolAlgo,
}

pub struct MklDnnPool2dFwd<T> {
  cfg:      MklDnnPool2dConfig,
  layout:   MklDnnLayout<T>,
  offsets:  Vec<i32>,
  inner:    dnnPrimitive_t,
  res:      Vec<*mut c_void>,
  src_layout:   MklDnnLayout<T>,
  dst_layout:   MklDnnLayout<T>,
  src_buf:      MklDnnBuffer<T>,
  dst_buf:      MklDnnBuffer<T>,
  workspace:    MklDnnBuffer<T>,
  load_src:     MklDnnConversion<T>,
  load_dst:     MklDnnConversion<T>,
  store_src:    MklDnnConversion<T>,
  store_dst:    MklDnnConversion<T>,
  _marker:  PhantomData<fn (T)>,
}

impl MklDnnPool2dFwd<f32> {
  pub fn create(cfg: MklDnnPool2dConfig) -> Result<MklDnnPool2dFwd<f32>, dnnError_t> {
    let mut inner: dnnPrimitive_t = null_mut();
    /*let in_strides = vec![
        1,
        cfg.in_dim[0],
        cfg.in_dim[0] * cfg.in_dim[1],
        cfg.in_dim[0] * cfg.in_dim[1] * cfg.in_dim[2],
    ];
    let mut layout = MklDnnLayout::create(cfg.in_dim.clone(), in_strides).unwrap();*/
    let layout = MklDnnLayout::create(cfg.in_dim.clone()).unwrap();
    let offsets = vec![-(cfg.pad[0] as i32), -(cfg.pad[1] as i32)];
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
    let src_layout = MklDnnLayout::create(cfg.in_dim.clone()).unwrap();
    let dst_layout = MklDnnLayout::create(cfg.out_dim.clone()).unwrap();
    //let src_buf = MklDnnBuffer::create(MklDnnLayout::create_from_primitive(inner, dnnResourceType_t::dnnResourceSrc).unwrap()).unwrap();
    let src_buf = MklDnnBuffer::create(MklDnnLayout::create(cfg.in_dim.clone()).unwrap()).unwrap();
    //let dst_buf = MklDnnBuffer::create(MklDnnLayout::create_from_primitive(inner, dnnResourceType_t::dnnResourceDst).unwrap()).unwrap();
    let dst_buf = MklDnnBuffer::create(MklDnnLayout::create(cfg.out_dim.clone()).unwrap()).unwrap();
    //let workspace = MklDnnBuffer::create(MklDnnLayout::create_from_primitive(inner, dnnResourceType_t::dnnResourceWorkspace).unwrap()).unwrap();
    let workspace = MklDnnBuffer::create(MklDnnLayout::create(cfg.out_dim.clone()).unwrap()).unwrap();
    let load_src = MklDnnConversion::create(&src_layout, &src_buf.layout).unwrap();
    let load_dst = MklDnnConversion::create(&dst_layout, &dst_buf.layout).unwrap();
    let store_src = MklDnnConversion::create(&src_buf.layout, &src_layout).unwrap();
    let store_dst = MklDnnConversion::create(&dst_buf.layout, &dst_layout).unwrap();
    Ok(MklDnnPool2dFwd{
      cfg:      cfg,
      layout:   layout,
      offsets:  offsets,
      inner:    inner,
      res:      res,
      src_layout:   src_layout,
      dst_layout:   dst_layout,
      src_buf:      src_buf,
      dst_buf:      dst_buf,
      workspace:    workspace,
      load_src:     load_src,
      load_dst:     load_dst,
      store_src:    store_src,
      store_dst:    store_dst,
      _marker:  PhantomData,
    })
  }

  pub fn _workspace(&self) -> *mut f32 {
    self.workspace.as_mut_ptr()
  }

  pub fn execute(&mut self, in_buf: *const f32, out_buf: *mut f32) -> Result<(), dnnError_t> {
    /*self.res[dnnResourceType_t::dnnResourceSrc as usize]        = in_buf as *mut _;
    self.res[dnnResourceType_t::dnnResourceDst as usize]        = out_buf as *mut _;*/

    self.res[dnnResourceType_t::dnnResourceSrc as usize]        = self.src_buf.as_mut_ptr() as *mut _;;
    self.res[dnnResourceType_t::dnnResourceDst as usize]        = self.dst_buf.as_mut_ptr() as *mut _;
    self.res[dnnResourceType_t::dnnResourceWorkspace as usize]  = self.workspace.as_mut_ptr() as *mut _;

    self.load_src.convert(in_buf, self.src_buf.as_mut_ptr()).unwrap();

    let status = unsafe { dnnExecute_F32(self.inner, self.res.as_mut_ptr()) };
    if status.is_err() {
      return Err(status);
    }

    self.store_dst.convert(self.dst_buf.as_ptr(), out_buf).unwrap();

    Ok(())
  }
}

pub struct MklDnnPool2dBwd<T> {
  cfg:      MklDnnPool2dConfig,
  layout:   MklDnnLayout<T>,
  offsets:  Vec<i32>,
  inner:    dnnPrimitive_t,
  res:      Vec<*mut c_void>,
  src_layout:   MklDnnLayout<T>,
  dst_layout:   MklDnnLayout<T>,
  src_buf:      MklDnnBuffer<T>,
  dst_buf:      MklDnnBuffer<T>,
  //workspace:    MklDnnBuffer<T>,
  load_src:     MklDnnConversion<T>,
  load_dst:     MklDnnConversion<T>,
  store_src:    MklDnnConversion<T>,
  store_dst:    MklDnnConversion<T>,
  _marker:  PhantomData<fn (T)>,
}

impl MklDnnPool2dBwd<f32> {
  pub fn create(cfg: MklDnnPool2dConfig) -> Result<MklDnnPool2dBwd<f32>, dnnError_t> {
    let mut inner: dnnPrimitive_t = null_mut();
    /*let in_strides = vec![
        1,
        cfg.in_dim[0],
        cfg.in_dim[0] * cfg.in_dim[1],
        cfg.in_dim[0] * cfg.in_dim[1] * cfg.in_dim[2],
    ];
    let layout = MklDnnLayout::create(cfg.in_dim.clone(), in_strides).unwrap();*/
    let layout = MklDnnLayout::create(cfg.in_dim.clone()).unwrap();
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
    let src_layout = MklDnnLayout::create(cfg.in_dim.clone()).unwrap();
    let dst_layout = MklDnnLayout::create(cfg.out_dim.clone()).unwrap();
    //let src_buf = MklDnnBuffer::create(MklDnnLayout::create_from_primitive(inner, dnnResourceType_t::dnnResourceDiffSrc).unwrap()).unwrap();
    let src_buf = MklDnnBuffer::create(MklDnnLayout::create(cfg.in_dim.clone()).unwrap()).unwrap();
    //let dst_buf = MklDnnBuffer::create(MklDnnLayout::create_from_primitive(inner, dnnResourceType_t::dnnResourceDst).unwrap()).unwrap();
    let dst_buf = MklDnnBuffer::create(MklDnnLayout::create(cfg.out_dim.clone()).unwrap()).unwrap();
    //let workspace = MklDnnBuffer::create(MklDnnLayout::create_from_primitive(inner, dnnResourceType_t::dnnResourceWorkspace).unwrap()).unwrap();
    let load_src = MklDnnConversion::create(&src_layout, &src_buf.layout).unwrap();
    let load_dst = MklDnnConversion::create(&dst_layout, &dst_buf.layout).unwrap();
    let store_src = MklDnnConversion::create(&src_buf.layout, &src_layout).unwrap();
    let store_dst = MklDnnConversion::create(&dst_buf.layout, &dst_layout).unwrap();
    Ok(MklDnnPool2dBwd{
      cfg:      cfg,
      layout:   layout,
      offsets:  offsets,
      inner:    inner,
      res:      res,
      src_layout:   src_layout,
      dst_layout:   dst_layout,
      src_buf:      src_buf,
      dst_buf:      dst_buf,
      //workspace:    workspace,
      load_src:     load_src,
      load_dst:     load_dst,
      store_src:    store_src,
      store_dst:    store_dst,
      _marker:  PhantomData,
    })
  }

  pub fn execute(&mut self, out_grad: *const f32, in_grad: *mut f32, workspace: *mut f32) -> Result<(), dnnError_t> {
    /*self.res[dnnResourceType_t::dnnResourceDiffDst as usize]    = out_grad as *mut _;
    self.res[dnnResourceType_t::dnnResourceDiffSrc as usize]    = in_grad as *mut _;*/

    self.res[dnnResourceType_t::dnnResourceDiffDst as usize]    = self.dst_buf.as_mut_ptr() as *mut _;
    self.res[dnnResourceType_t::dnnResourceDiffSrc as usize]    = self.src_buf.as_mut_ptr() as *mut _;;
    //self.res[dnnResourceType_t::dnnResourceWorkspace as usize]  = self.workspace.as_mut_ptr() as *mut _;
    self.res[dnnResourceType_t::dnnResourceWorkspace as usize]  = workspace as *mut _;


    self.load_dst.convert(out_grad, self.dst_buf.as_mut_ptr()).unwrap();

    let status = unsafe { dnnExecute_F32(self.inner, self.res.as_mut_ptr()) };
    if status.is_err() {
      return Err(status);
    }

    self.store_src.convert(self.src_buf.as_ptr(), in_grad).unwrap();

    Ok(())
  }
}
