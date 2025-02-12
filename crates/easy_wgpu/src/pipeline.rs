use crate::bind_group_layout::BindGroupLayoutDesc;
// use anyhow::Context as _;
use smallvec::SmallVec;
use std::borrow::Cow;
use wgpu;

pub const DEFAULT_VS_SHADER_ENTRY_POINT: &str = "vs_main";
pub const DEFAULT_FS_SHADER_ENTRY_POINT: &str = "fs_main";

/// Creates a render pipeline descriptor, can then be used to create a concrete
/// pipeline or for later for using the descriptor as a hash key in a pipeline
/// arena `RenderPipelineDesc` can be converted into [`wgpu::RenderPipeline`]
/// (which isn't hashable or comparable)
#[derive(Clone, Hash, PartialEq, Eq)]
pub struct RenderPipelineDesc {
    pub label: Option<String>,
    // pub shader_path: String,
    pub shader_code: Option<String>, //single file for both vertex and fragment
    pub shader_code_vert: Option<String>,
    pub shader_code_frag: Option<String>,
    pub shader_label: Option<String>,
    pub disable_fragment_shader: bool, //fragment state can be none when rendering to shadow map
    //layout for each group of bindings in the pipline
    pub bind_group_layouts_desc: SmallVec<[BindGroupLayoutDesc; 4]>,
    /// The format of any vertex buffers used with this pipeline.
    pub vertex_buffers_layouts: SmallVec<[wgpu::VertexBufferLayout<'static>; 4]>,
    /// The color state of the render targets.
    pub render_targets: SmallVec<[Option<wgpu::ColorTargetState>; 4]>,

    //most of the stuff after this are usually default
    /// The properties of the pipeline at the primitive assembly and
    /// rasterization level.
    pub primitive: wgpu::PrimitiveState,
    /// The effect of draw calls on the depth and stencil aspects of the output
    /// target, if any.
    pub depth_stencil: Option<wgpu::DepthStencilState>,
    /// The multi-sampling properties of the pipeline.
    pub multisample: wgpu::MultisampleState,
}

impl Default for RenderPipelineDesc {
    fn default() -> Self {
        Self {
            label: None,
            // pub shader_path: String,
            shader_code: None,
            shader_code_vert: None,
            shader_code_frag: None,
            shader_label: Some(String::from("Shader")),
            disable_fragment_shader: false,
            //layout for each group of bindings in the pipline
            bind_group_layouts_desc: SmallVec::new(),
            // The format of any vertex buffers used with this pipeline.
            vertex_buffers_layouts: SmallVec::new(),
            // The color state of the render targets.
            render_targets: SmallVec::new(),

            //most of the stuff after this are usually default
            // The properties of the pipeline at the primitive assembly and rasterization level.
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                // cull_mode: Some(wgpu::Face::Back),
                cull_mode: None, //LEAVE it here because we want to visualize the backside of the planes used for Lights
                // Setting this to anything other than Fill requires Features::POLYGON_MODE_LINE
                // or Features::POLYGON_MODE_POINT
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            // The effect of draw calls on the depth and stencil aspects of the output target, if any.
            depth_stencil: None,
            // The multi-sampling properties of the pipeline.
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
        }
    }
}
impl RenderPipelineDesc {
    /// # Panics
    /// Will panic if no shader was set
    pub fn into_render_pipeline(
        self,
        device: &wgpu::Device,
        // ) -> wgpu::RenderPipelineDescriptor<'a> {
    ) -> wgpu::RenderPipeline {
        //load shader
        //we declare the shader here because we want it to live long enough so we can
        // have reference to it
        let shader_monolithic: Option<wgpu::ShaderModule>;
        let shader_vert: Option<wgpu::ShaderModule>;
        let shader_frag: Option<wgpu::ShaderModule>;
        let (shader_vert, shader_frag) = if let Some(shader_code_monlithic) = self.shader_code {
            shader_monolithic = Some(device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: self.shader_label.as_deref(),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&shader_code_monlithic)),
            }));
            (shader_monolithic.as_ref().unwrap(), shader_monolithic.as_ref().unwrap())
        } else {
            shader_vert = Some(device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: self.shader_label.as_deref(),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(self.shader_code_vert.as_ref().unwrap())),
            }));
            shader_frag = Some(device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: self.shader_label.as_deref(),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(self.shader_code_frag.as_ref().unwrap())),
            }));
            (shader_vert.as_ref().unwrap(), shader_frag.as_ref().unwrap())
        };

        // let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        //     label: self.shader_label.as_deref(),
        //     source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&self.shader_code)),
        // });

        // //make the bind group layouts
        let mut bind_group_layouts: Vec<wgpu::BindGroupLayout> = Vec::new();
        // for bgl_desc in self.pipeline_layout_desc.bind_group_layouts_desc {
        for bgl_desc in self.bind_group_layouts_desc {
            // let bgl = device
            // .create_bind_group_layout(&bgl_desc.
            // into_bind_group_layout_descriptor(device));
            let bgl = bgl_desc.into_bind_group_layout(device);
            bind_group_layouts.push(bgl);
        }

        //from &[T] to &[&T] because the bind group layouts expects that
        let bind_group_layouts: Vec<&wgpu::BindGroupLayout> = bind_group_layouts.iter().collect();

        // //make a render pipeline layout
        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: bind_group_layouts.as_slice(),
            push_constant_ranges: &[],
        });

        //fragment state can be none when rendering to shadow map
        let fragment_state: Option<wgpu::FragmentState> = if self.disable_fragment_shader {
            None
        } else {
            Some(wgpu::FragmentState {
                module: shader_frag,
                entry_point: DEFAULT_FS_SHADER_ENTRY_POINT,
                targets: self.render_targets.as_slice(),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            })
        };

        let pipeline_desc = wgpu::RenderPipelineDescriptor {
            label: self.label.as_deref(),
            // layout: Some(&render_pipeline_layout),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: shader_vert,
                entry_point: DEFAULT_VS_SHADER_ENTRY_POINT,
                buffers: self.vertex_buffers_layouts.as_slice(),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: fragment_state,
            primitive: self.primitive,
            depth_stencil: self.depth_stencil,
            multisample: self.multisample,
            multiview: None,
            cache: None,
        };

        // pipeline_desc
        device.create_render_pipeline(&pipeline_desc)
    }
}

/// A builder type to help simplify the construction of a `RenderPipelineDesc`.
/// We've attempted to provide a suite of reasonable defaults in the case that
/// none are provided.
pub struct RenderPipelineDescBuilder {
    pipeline_desc: Option<RenderPipelineDesc>,
}
impl Default for RenderPipelineDescBuilder {
    fn default() -> Self {
        Self::new()
    }
}
impl RenderPipelineDescBuilder {
    pub fn new() -> Self {
        Self {
            pipeline_desc: Some(RenderPipelineDesc::default()),
        }
    }

    /// # Panics
    /// Will panic if the builder was not constructed with ``new()``
    #[must_use]
    pub fn label(mut self, label: &str) -> Self {
        self.pipeline_desc.as_mut().unwrap().label = Some(String::from(label));
        self
    }

    /// # Panics
    /// Will panic if the builder was not constructed with ``new()``
    #[must_use]
    pub fn shader_code(mut self, code: &str) -> Self {
        self.pipeline_desc.as_mut().unwrap().shader_code = Some(String::from(code));
        self
    }

    /// # Panics
    /// Will panic if the builder was not constructed with ``new()``
    #[must_use]
    pub fn shader_code_vert(mut self, code: &str) -> Self {
        self.pipeline_desc.as_mut().unwrap().shader_code_vert = Some(String::from(code));
        self
    }

    /// # Panics
    /// Will panic if the builder was not constructed with ``new()``
    #[must_use]
    pub fn shader_code_frag(mut self, code: &str) -> Self {
        self.pipeline_desc.as_mut().unwrap().shader_code_frag = Some(String::from(code));
        self
    }

    /// # Panics
    /// Will panic if the builder was not constructed with ``new()``
    #[must_use]
    pub fn shader_label(mut self, label: &str) -> Self {
        self.pipeline_desc.as_mut().unwrap().shader_label = Some(String::from(label));
        self
    }

    /// # Panics
    /// Will panic if the builder was not constructed with ``new()``
    #[must_use]
    pub fn disable_fragment_shader(mut self) -> Self {
        self.pipeline_desc.as_mut().unwrap().disable_fragment_shader = true;
        self
    }

    /// # Panics
    /// Will panic if the builder was not constructed with ``new()``
    #[must_use]
    pub fn add_bind_group_layout_desc(mut self, layout_desc: BindGroupLayoutDesc) -> Self {
        self.pipeline_desc.as_mut().unwrap().bind_group_layouts_desc.push(layout_desc);
        self
    }

    /// # Panics
    /// Will panic if the builder was not constructed with ``new()``
    #[must_use]
    pub fn add_vertex_buffer_layout(mut self, vertex_layout: wgpu::VertexBufferLayout<'static>) -> Self {
        self.pipeline_desc.as_mut().unwrap().vertex_buffers_layouts.push(vertex_layout);
        self
    }

    /// # Panics
    /// Will panic if the builder was not constructed with ``new()``
    #[must_use]
    pub fn add_render_target(mut self, render_target: wgpu::ColorTargetState) -> Self {
        self.pipeline_desc.as_mut().unwrap().render_targets.push(Some(render_target));
        self
    }

    /// # Panics
    /// Will panic if the builder was not constructed with ``new()``
    #[must_use]
    pub fn primitive(mut self, primitive: wgpu::PrimitiveState) -> Self {
        self.pipeline_desc.as_mut().unwrap().primitive = primitive;
        self
    }

    /// # Panics
    /// Will panic if the builder was not constructed with ``new()``
    #[must_use]
    pub fn depth_state(mut self, depth_state: Option<wgpu::DepthStencilState>) -> Self {
        self.pipeline_desc.as_mut().unwrap().depth_stencil = depth_state;
        self
    }

    /// # Panics
    /// Will panic if the builder was not constructed with ``new()``
    #[must_use]
    pub fn multisample(mut self, multisample: wgpu::MultisampleState) -> Self {
        self.pipeline_desc.as_mut().unwrap().multisample = multisample;
        self
    }

    /// # Panics
    /// Will panic if the builder was not constructed with ``new()``
    pub fn build_desc(&mut self) -> RenderPipelineDesc {
        self.pipeline_desc.take().unwrap() //move out of the builder and
                                           // returns a fully fledged object
    }

    /// # Panics
    /// Will panic if the builder was not constructed with ``new()``
    pub fn build_pipeline(&mut self, device: &wgpu::Device) -> wgpu::RenderPipeline {
        let desc = self.pipeline_desc.take().unwrap();
        desc.into_render_pipeline(device)
    }
}
