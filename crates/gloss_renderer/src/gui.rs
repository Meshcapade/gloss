use crate::{
    components::{
        Colors, DiffuseImg, DiffuseTex, Faces, ImgConfig, LightEmit, MeshColorType, ModelMatrix, Name, NormalImg, NormalTex, Normals, PointColorType,
        PosLookat, Projection, Renderable, RoughnessImg, RoughnessTex, ShadowCaster, ShadowMapDirty, UVs, Verts, VisLines, VisMesh, VisNormals,
        VisPoints, VisWireframe,
    },
    config::Config,
    viewer::Runner,
};

use crate::plugin_manager::gui::window::{GuiWindowType, WindowPivot, WindowPositionType};

use crate::{
    forward_renderer::Renderer,
    geom::Geom,
    plugin_manager::plugins::Plugins,
    scene::{Scene, GLOSS_FLOOR_NAME},
};
use utils_rs::abi_stable_aliases::std_types::{ROption::RSome, RString, RVec};

use egui::style::TextCursorStyle;
use log::debug;
use utils_rs::string::float2string;

use easy_wgpu::gpu::Gpu;
use egui_wgpu::ScreenDescriptor;

use egui::style::{HandleShape, NumericColorSpace, ScrollStyle};

use gloss_memory::{accounting_allocator, CallstackStatistics, MemoryUse};
use utils_rs::tensor::DynamicMatrixOps;

use log::error;
use winit::window::Window;

use crate::plugin_manager::gui::widgets::Widgets as WidgetsFFI;
use egui::{
    epaint,
    epaint::Shadow,
    scroll_area,
    style::{Interaction, Selection, Spacing, WidgetVisuals, Widgets},
    Align, Align2, Color32, FontId, Layout, RichText, Rounding, ScrollArea, Slider, Stroke, Style, Ui, Vec2, Visuals,
};
use egui_winit::{self, EventResponse};
use epaint::Margin;

use gloss_hecs::{CommandBuffer, Entity};

use std::{collections::HashMap, path::Path};

use nalgebra as na;

// check the integration example here: https://docs.rs/egui/latest/egui/
// info of other people trying to do custom stuff: https://users.rust-lang.org/t/egui-is-it-possible-to-avoid-using-eframe/70470/22
// some other example of a large codebase using egui: https://github.com/parasyte/cartunes
// example of egui-winit and egui-wgpu: https://github.com/hasenbanck/egui_example/blob/master/src/main.rs
// official example of egui-wgpu: https://github.com/emilk/egui/blob/master/crates/egui_demo_app/src/apps/custom3d_wgpu.rs

// integration
const SIDE_PANEL_WIDTH: f32 = 180.0;
const SPACING_1: f32 = 10.0;

/// Separate the egui ctx from the rest of the gui because borrow checker
/// complains when we modify state mutably of the gui and also have immutable
/// reference to `egui_ctx`. having a mutable widget the deal only with state
/// solved this
pub struct GuiMainWidget {
    //contains the gui state
    pub selected_mesh_name: String,
    pub selected_entity: Option<Entity>,
    pub selected_light_name: String,
    pub selected_light_entity: Option<Entity>,
    pub wgputex_2_eguitex: HashMap<wgpu::Id<wgpu::Texture>, epaint::TextureId>,
    pub hovered_diffuse_tex: bool,
    pub hovered_normal_tex: bool,
    pub hovered_roughness_tex: bool,
    default_texture: Option<easy_wgpu::texture::Texture>,
    //gizmo stuff
    // gizmo_mode: GizmoMode,
    // gizmo_orientation: GizmoOrientation,
}
impl Default for GuiMainWidget {
    #[allow(clippy::derivable_impls)]
    fn default() -> GuiMainWidget {
        GuiMainWidget {
            selected_mesh_name: String::new(),
            selected_light_name: String::new(),
            selected_entity: None,
            selected_light_entity: None,
            wgputex_2_eguitex: HashMap::new(),
            hovered_diffuse_tex: false,
            hovered_normal_tex: false,
            hovered_roughness_tex: false,
            default_texture: None,
            // gizmo_mode: GizmoMode::Translate,
            // gizmo_orientation: GizmoOrientation::Local,
        }
    }
}
impl GuiMainWidget {
    pub fn new(gpu: &Gpu) -> Self {
        let path_tex = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/uv_checker.png");
        debug!("path_tex {path_tex}");
        let default_texture = easy_wgpu::texture::Texture::create_default_texture(gpu.device(), gpu.queue());

        Self {
            default_texture: Some(default_texture),
            ..Default::default()
        }
    }
}

type CbFnType = fn(&mut GuiMainWidget, ctx: &egui::Context, ui: &mut Ui, renderer: &Renderer, scene: &mut Scene);
pub struct Gui {
    egui_ctx: egui::Context,           //we do all the gui rendering inside this context
    pub egui_state: egui_winit::State, //integrator with winit https://github.com/emilk/egui/blob/master/crates/egui-winit/src/lib.rs#L55
    //similar to voiding on github
    egui_renderer: egui_wgpu::Renderer,
    width: u32,
    height: u32,
    pub hidden: bool,
    gui_main_widget: GuiMainWidget,
    command_buffer: CommandBuffer, //defer insertions and deletion of scene entities for whenever we apply this command buffer
    //callbacks
    //https://users.rust-lang.org/t/callback-with-generic/52426/5
    //https://stackoverflow.com/questions/66832392/sending-method-as-callback-function-to-field-object-in-rust
    //https://stackoverflow.com/questions/41081240/idiomatic-callbacks-in-rust
    //https://www.reddit.com/r/rust/comments/gi2pld/callback_functions_the_right_way/
    //https://github.com/rhaiscript/rhai/issues/178
    //https://www.reddit.com/r/rust/comments/ymingb/what_is_the_idiomatic_approach_to_eventscallbacks/iv5pgz9/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button

    //callback function that can add gui elements either inside the sidebar or outside the sidebar
    callbacks: Vec<CbFnType>,
    callbacks_for_selected_mesh: Vec<CbFnType>,
}

impl Gui {
    pub fn new(window: &winit::window::Window, gpu: &Gpu, surface_format: wgpu::TextureFormat) -> Self {
        #[allow(clippy::cast_possible_truncation)] //it's ok, we don't have very big numbers
        let native_pixels_per_point = window.scale_factor() as f32;

        let egui_renderer = egui_wgpu::Renderer::new(gpu.device(), surface_format, None, 1, false);

        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(), //just a shallow clone since it's behind an Arc
            egui::ViewportId::default(),
            window,
            Some(native_pixels_per_point),
            None,
            Some(2048), //TODO maybe find the concrete value, for now we leave 2048 because wasm
        ); //state that gets all the events from the window and gather them
           //https://github.com/emilk/egui/blob/bdc8795b0476c25faab927fc3c731f2d79f2098f/crates/eframe/src/native/epi_integration.rs#L361

        //size of the gui window. Will get resized automatically
        let width = 100;
        let height = 100;

        //Gui state based on what the user does
        let gui_main_widget = GuiMainWidget::new(gpu);

        // Mutate global style with above changes
        #[allow(unused_mut)]
        let mut style = style();

        //on web the view only renders when there is a mouse being dragged or when
        // there is an input so animations are choppy if you don't move the mouse.
        // Therefore we disable them
        cfg_if::cfg_if! {
            if #[cfg(target_arch = "wasm32")] {
                style.animation_time = 0.0;
            }
        }
        egui_ctx.set_style(style);

        let command_buffer = CommandBuffer::new();

        Self {
            egui_ctx,
            egui_state,
            egui_renderer,
            width,
            height,
            hidden: false,
            gui_main_widget,
            command_buffer,
            callbacks: Vec::new(),
            callbacks_for_selected_mesh: Vec::new(),
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }

    //TODO rename to "process gui event" for coherency with the other event
    // processing things
    pub fn on_event(&mut self, window: &Window, event: &winit::event::WindowEvent) -> EventResponse {
        self.egui_state.on_window_event(window, event)
    }

    /// # Panics
    /// Will panic is the path is not valid unicode
    pub fn on_drop(&mut self, path_buf: &Path, scene: &mut Scene) {
        self.gui_main_widget.set_default_selected_entity(scene);

        let path = path_buf.to_str().unwrap();
        let entity = self.gui_main_widget.selected_entity.unwrap();
        if self.gui_main_widget.hovered_diffuse_tex {
            scene
                .world
                .insert_one(entity, DiffuseImg::new_from_path(path, &ImgConfig::default()))
                .ok();
        }
        if self.gui_main_widget.hovered_normal_tex {
            scene.world.insert_one(entity, NormalImg::new_from_path(path, &ImgConfig::default())).ok();
        }
        if self.gui_main_widget.hovered_roughness_tex {
            scene
                .world
                .insert_one(entity, RoughnessImg::new_from_path(path, &ImgConfig::default()))
                .ok();
        }
    }

    pub fn wants_pointer_input(&self) -> bool {
        //tryng to solve https://github.com/urholaukkarinen/egui-gizmo/issues/19
        self.egui_ctx.wants_pointer_input()
    }

    pub fn is_hovering(&self) -> bool {
        self.egui_ctx.is_pointer_over_area()
    }

    //inspiration from voidin renderer on github
    //https://github.com/pudnax/voidin/blob/91e6b564008879388f3777bcb6154c656bfc533c/crates/app/src/app.rs#L643
    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &mut self,
        window: &winit::window::Window,
        gpu: &Gpu,
        renderer: &Renderer,
        runner: &Runner,
        scene: &mut Scene,
        plugins: &Plugins,
        config: &mut Config,
        out_view: &wgpu::TextureView,
    ) {
        if self.hidden {
            return;
        }
        self.begin_frame();

        let screen_descriptor = ScreenDescriptor {
            size_in_pixels: [self.width, self.height],
            pixels_per_point: self.egui_ctx.pixels_per_point(),
        };

        let full_output = self.egui_ctx.run(self.egui_state.take_egui_input(window), |ctx: &egui::Context| {
            // ui_builder(ctx) // THIS ACTUALLY RENDERS GUI
            self.gui_main_widget.build_gui(
                self.width,
                self.height,
                ctx,
                renderer,
                &mut self.egui_renderer,
                gpu,
                scene,
                config,
                runner,
                &mut self.command_buffer,
                &mut self.callbacks,
                &mut self.callbacks_for_selected_mesh,
                plugins,
            );
        });
        let paint_jobs = self.egui_ctx.tessellate(full_output.shapes, self.egui_ctx.pixels_per_point());
        let textures_delta = full_output.textures_delta;

        let mut encoder = gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Gui") });
        {
            for (texture_id, image_delta) in &textures_delta.set {
                self.egui_renderer.update_texture(gpu.device(), gpu.queue(), *texture_id, image_delta);
            }
            for texture_id in &textures_delta.free {
                self.egui_renderer.free_texture(texture_id);
            }
            self.egui_renderer
                .update_buffers(gpu.device(), gpu.queue(), &mut encoder, &paint_jobs, &screen_descriptor);

            let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("UI Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: out_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            self.egui_renderer
                .render(&mut render_pass.forget_lifetime(), paint_jobs.as_slice(), &screen_descriptor);
        }
        gpu.queue().submit(Some(encoder.finish()));
        self.end_frame(scene);
    }

    fn begin_frame(&self) {}

    fn end_frame(&mut self, scene: &mut Scene) {
        self.command_buffer.run_on(&mut scene.world);
    }

    pub fn add_callback(
        &mut self,
        f: fn(&mut GuiMainWidget, ctx: &egui::Context, ui: &mut Ui, renderer: &Renderer, scene: &mut Scene),
        draw_in_global_panel: bool,
    ) {
        if draw_in_global_panel {
            self.callbacks.push(f);
        } else {
            self.callbacks_for_selected_mesh.push(f);
        }
    }
}

impl GuiMainWidget {
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_lines)]
    fn build_gui(
        &mut self,
        screen_width: u32,
        screen_height: u32,
        ctx: &egui::Context,
        renderer: &Renderer,
        egui_renderer: &mut egui_wgpu::Renderer,
        gpu: &Gpu,
        scene: &mut Scene,
        config: &mut Config,
        runner: &Runner,
        command_buffer: &mut CommandBuffer,
        callbacks: &mut [CbFnType],
        callbacks_for_selected_mesh: &mut [CbFnType],
        plugins: &Plugins,
    ) {
        self.set_default_selected_entity(scene);

        //draw point indices for the selected entity
        if let Some(ent) = self.selected_entity {
            if let Ok(mut c) = scene.get_comp::<&mut VisPoints>(&ent) {
                self.draw_verts_indices(ctx, scene, screen_width, screen_height, &mut c);
            }
        }

        egui::SidePanel::left("my_left_panel").default_width(SIDE_PANEL_WIDTH).show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                //Scene
                egui::CollapsingHeader::new("Scene").show(ui, |ui| {
                    ui.group(|ui| {
                        ScrollArea::vertical()
                            .max_height(200.0)
                            .scroll_bar_visibility(scroll_area::ScrollBarVisibility::AlwaysVisible)
                            .auto_shrink([false, false])
                            .show(ui, |ui| {
                                ui.with_layout(Layout::top_down_justified(Align::LEFT), |ui| {
                                    ui.spacing_mut().item_spacing.y = 0.0;
                                    ui.spacing_mut().button_padding.y = 4.0;

                                    //get all entities that are renderable and sort by name
                                    let entities = scene.get_renderables(true);

                                    //go through all visible meshes and show their name
                                    // for (_cur_mesh_idx, e_ref) in scene.world.iter().enumerate() {
                                    for entity in entities {
                                        let e_ref = scene.world.entity(entity).unwrap();
                                        //get the name of the mesh which acts like a unique id
                                        let name = e_ref.get::<&Name>().expect("The entity has no name").0.clone();

                                        //GUI for this concrete mesh
                                        //if we click we can see options for vis
                                        let _res = ui.selectable_value(&mut self.selected_mesh_name, name.clone(), &name);

                                        if name == self.selected_mesh_name {
                                            self.selected_entity = Some(entity);
                                            //make a side window
                                            self.draw_vis(ctx, renderer, scene, entity, command_buffer, callbacks_for_selected_mesh);
                                        }
                                    }
                                });
                            });
                    });
                });

                // Params
                egui::CollapsingHeader::new("Textures").show(ui, |ui| {
                    self.draw_textures(ui, scene, egui_renderer, gpu, command_buffer);
                });

                //Move
                // egui::CollapsingHeader::new("Move")
                // .show(ui, |ui| self.draw_move(ui, scene, command_buffer));

                // Params
                egui::CollapsingHeader::new("Params").show(ui, |ui| {
                    self.draw_params(ui, scene, config, command_buffer);
                });

                // Lights
                egui::CollapsingHeader::new("Lights").show(ui, |ui| self.draw_lights(ui, scene, command_buffer));

                // Cam
                egui::CollapsingHeader::new("Camera").show(ui, |ui| self.draw_cam(ui, scene, command_buffer));

                // Io
                egui::CollapsingHeader::new("Io").show(ui, |ui| {
                    self.draw_io(ui, scene, command_buffer, self.selected_entity);
                });

                // profiling
                egui::CollapsingHeader::new("Profiling").show(ui, |ui| self.draw_profiling(ui, scene, command_buffer));

                // Plugins
                egui::CollapsingHeader::new("Plugins").show(ui, |ui| {
                    self.draw_plugins(ui, scene, plugins, command_buffer);
                });

                //fps
                ui.separator();
                let dt = runner.dt();
                let fps = 1.0 / dt.as_secs_f32();
                let ms = dt.as_millis();
                let fps_string = format!("{fps:.0}");
                let ms_string = format!("{ms:.2}");
                ui.label(egui::RichText::new("FPS: ".to_owned() + &fps_string));
                ui.label(egui::RichText::new("dt(ms): ".to_owned() + &ms_string));

                // A `scope` creates a temporary [`Ui`] in which you can change settings:
                // TODO: Change wrap mode?
                ui.scope(|ui| {
                    ui.visuals_mut().override_text_color = Some(egui::Color32::RED);
                    ui.style_mut().override_text_style = Some(egui::TextStyle::Monospace);
                    ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Wrap);
                });

                ui.add(egui::Separator::default());
                for f in callbacks.iter() {
                    f(self, ctx, ui, renderer, scene);
                }

                for system_and_metadata in plugins.gui_systems.iter() {
                    let sys = &system_and_metadata.0;
                    let func = sys.f;
                    let gui_window = func(self.selected_entity.into(), scene);
                    let window_name = gui_window.window_name;
                    let widgets = gui_window.widgets;
                    let window_type = gui_window.window_type;

                    if widgets.is_empty() {
                        continue; //there's no widgets so there's nothing to
                                  // draw
                    }

                    //recursivelly draw all widgets
                    // https://stackoverflow.com/a/72862424
                    let mut draw_widgets = |ui: &mut Ui| {
                        //we make a helper so that we can call recursivelly
                        fn helper(ui: &mut Ui, widgets: &RVec<WidgetsFFI>, selected_entity: Entity, scene: &mut Scene) {
                            for widget in widgets.iter() {
                                match widget {
                                    WidgetsFFI::Slider(slider) => {
                                        let mut val = slider.init_val;
                                        if let RSome(slider_width) = slider.width {
                                            ui.spacing_mut().slider_width = slider_width;
                                            //changes size od slider2
                                        }
                                        let res = ui.add(
                                            Slider::new(&mut val, slider.min..=slider.max)
                                                .fixed_decimals(3)
                                                .text(slider.name.as_str()),
                                        );
                                        if res.dragged() {
                                            (slider.f_change)(val, slider.name.clone(), selected_entity, scene);
                                        }
                                        // } else {
                                        if res.drag_stopped() {
                                            // Updated method here
                                            if let RSome(func) = slider.f_no_change {
                                                func(slider.name.clone(), selected_entity, scene);
                                            }
                                        }
                                        // if res.drag_released() {
                                        //     if let RSome(func) =
                                        // slider.f_no_change {
                                        //         func(slider.name.clone(),
                                        // selected_entity, scene);
                                        //     }
                                        // }
                                    }
                                    WidgetsFFI::Checkbox(checkbox) => {
                                        let mut val = checkbox.init_val;
                                        let res = ui.add(egui::Checkbox::new(&mut val, checkbox.name.as_str()));
                                        if res.clicked() {
                                            (checkbox.f_clicked)(val, checkbox.name.clone(), selected_entity, scene);
                                        }
                                    }
                                    WidgetsFFI::Button(button) => {
                                        if ui.add(egui::Button::new(button.name.as_str())).clicked() {
                                            (button.f_clicked)(button.name.clone(), selected_entity, scene);
                                        }
                                    }
                                    WidgetsFFI::SelectableList(selectable_list) => {
                                        let mut draw_selectables = |ui: &mut Ui| {
                                            for item in selectable_list.items.iter() {
                                                if ui.add(egui::SelectableLabel::new(item.is_selected, item.name.to_string())).clicked() {
                                                    (item.f_clicked)(item.name.clone(), selected_entity, scene);
                                                }
                                            }
                                        };

                                        if selectable_list.is_horizontal {
                                            ui.horizontal(draw_selectables);
                                        } else {
                                            draw_selectables(ui);
                                        }
                                    }
                                    WidgetsFFI::Horizontal(widgets) => {
                                        ui.horizontal(|ui| {
                                            helper(ui, widgets, selected_entity, scene);
                                        });
                                    }
                                }
                            }
                        }
                        if let Some(selected_entity) = self.selected_entity {
                            //finally call the helper function so that we start the recursion
                            helper(ui, &widgets, selected_entity, scene);
                        }
                    };

                    match window_type {
                        #[allow(clippy::cast_precision_loss)]
                        GuiWindowType::FloatWindow(pivot, position, position_type) => {
                            // egui::Window::new(window_name.to_string()).show(ctx, &mut draw_widgets);
                            let pos_x = (screen_width as f32 - SIDE_PANEL_WIDTH) * position.0[0];
                            let pos_y = (screen_height as f32) * position.0[1];
                            let pivot = match pivot {
                                WindowPivot::LeftBottom => Align2::LEFT_BOTTOM,
                                WindowPivot::LeftCenter => Align2::LEFT_CENTER,
                                WindowPivot::LeftTop => Align2::LEFT_TOP,
                                WindowPivot::CenterBottom => Align2::CENTER_BOTTOM,
                                WindowPivot::CenterCenter => Align2::CENTER_CENTER,
                                WindowPivot::CenterTop => Align2::CENTER_TOP,
                                WindowPivot::RightBottom => Align2::RIGHT_BOTTOM,
                                WindowPivot::RightCenter => Align2::RIGHT_CENTER,
                                WindowPivot::RightTop => Align2::RIGHT_TOP,
                            };

                            let mut win = egui::Window::new(window_name.to_string()).pivot(pivot);
                            match position_type {
                                WindowPositionType::Fixed => {
                                    win = win.fixed_pos([pos_x, pos_y]);
                                }
                                WindowPositionType::Initial => {
                                    win = win.default_pos([pos_x, pos_y]);
                                }
                            }
                            // win.show(ctx, &mut draw_widgets);
                            win.show(ctx, &mut draw_widgets);
                        }
                        GuiWindowType::Sidebar => {
                            egui::CollapsingHeader::new(window_name.to_string()).show(ui, &mut draw_widgets);
                        }
                    };
                }
            });
        });
    }

    // if no selected mesh is set yet, any query with the name will fail.
    // we can use this function to set it to some default value(usually the first
    // mesh in my list)
    fn set_default_selected_entity(&mut self, scene: &Scene) {
        for e_ref in scene.world.iter() {
            // let entity = e_ref.entity();
            let is_renderable = e_ref.has::<Renderable>();
            if is_renderable {
                //get the name of the mesh which acts like a unique id
                let name = e_ref.get::<&Name>().expect("The entity has no name").0.clone();

                //if it's the first time we encounter a renderable mesh,  we set the selected
                // name to this one
                if self.selected_mesh_name.is_empty() && name != GLOSS_FLOOR_NAME {
                    self.selected_mesh_name.clone_from(&name);
                    self.selected_entity = Some(e_ref.entity());
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_vis(
        &mut self,
        ctx: &egui::Context,
        renderer: &Renderer,
        scene: &mut Scene,
        entity: Entity,
        command_buffer: &mut CommandBuffer,
        callbacks_for_selected_mesh: &mut [CbFnType],
    ) {
        let e_ref = scene.world.entity(entity).unwrap();
        let has_vis_points = e_ref.has::<VisPoints>();
        let has_vis_lines = e_ref.has::<VisLines>();
        let _has_vis_wireframe = e_ref.has::<VisWireframe>();
        let has_vis_mesh = e_ref.has::<VisMesh>();
        let _has_vis_normals = e_ref.has::<VisNormals>();
        let mut _window = egui::Window::new("vis_points")
            // .auto_sized()
            .default_width(100.0)
            // .min_height(600.0)
            .resizable(false)
            // .collapsible(true)
            .title_bar(false)
            .scroll([false, false])
            .anchor(Align2::LEFT_TOP, Vec2::new(SIDE_PANEL_WIDTH + 12.0, 0.0))
            .show(ctx, |ui| {
                //dummy vis options that we use to draw invisible widgets
                //we need this because when we don't have a component we still want to draw an
                // empty space for it and we use this dummy widget to figure out how much space
                // we need points
                if has_vis_points {
                    ui.add_space(SPACING_1);
                    let mut c = scene.get_comp::<&mut VisPoints>(&entity).unwrap();
                    self.draw_vis_points(ui, scene, entity, command_buffer, has_vis_points, &mut c);
                }
                //mesh
                if has_vis_mesh {
                    ui.add_space(SPACING_1);
                    // let mut c = scene.get_comp::<&mut VisMesh>(&entity);
                    self.draw_vis_mesh(ui, scene, entity, command_buffer, has_vis_mesh);
                }
                //lines
                if has_vis_lines {
                    ui.add_space(SPACING_1);
                    let mut c = scene.get_comp::<&mut VisLines>(&entity).unwrap();
                    self.draw_vis_lines(ui, scene, entity, command_buffer, has_vis_lines, &mut c);
                }
                // TODO: Keep this?
                //wireframe
                // if has_vis_wireframe {
                //     ui.add_space(SPACING_1);
                //     let mut c = scene.get_comp::<&mut VisWireframe>(&entity);
                //     self.draw_vis_wireframe(
                //         // ctx,
                //         ui,
                //         // renderer,
                //         scene,
                //         entity,
                //         command_buffer,
                //         has_vis_wireframe,
                //         &mut c,
                //     );
                // }
                //normals
                // if has_vis_normals {
                //     ui.add_space(SPACING_1);
                //     let mut c = scene.get_comp::<&mut VisNormals>(&entity);
                //     self.draw_vis_normals(
                //         // ctx,
                //         ui,
                //         // renderer,
                //         scene,
                //         entity,
                //         command_buffer,
                //         has_vis_wireframe,
                //         &mut c,
                //     );
                // }

                self.draw_comps(ui, scene, entity, command_buffer, true);

                for f in callbacks_for_selected_mesh.iter() {
                    f(self, ctx, ui, renderer, scene);
                }
            });
    }

    #[allow(clippy::cast_precision_loss)]
    fn draw_verts_indices(&self, ctx: &egui::Context, scene: &Scene, screen_width: u32, screen_height: u32, c: &mut VisPoints) {
        if !c.show_points_indices {
            return;
        }
        //TODO remove all these unwraps
        egui::Area::new(egui::Id::new("verts_indices"))
            .interactable(false)
            .anchor(Align2::LEFT_TOP, egui::vec2(0.0, 0.0))
            .show(ctx, |ui| {
                if let Some(ent) = self.selected_entity {
                    let verts = scene.get_comp::<&Verts>(&ent).unwrap();
                    let model_matrix = scene.get_comp::<&ModelMatrix>(&ent).unwrap();
                    let cam = scene.get_current_cam().unwrap();
                    let view = cam.view_matrix(scene);
                    let proj = cam.proj_matrix(scene);
                    for (idx, vert) in verts.0.to_dmatrix().row_iter().enumerate() {
                        let point_world = model_matrix.0 * na::Point3::from(vert.fixed_columns::<3>(0).transpose());
                        let point_screen = cam.project(
                            point_world,
                            view,
                            proj,
                            na::Vector2::<f32>::new(screen_width as f32, screen_height as f32),
                        );

                        let widget_max_size = egui::vec2(35.0, 35.0);
                        let widget_rect = egui::Rect::from_min_size(
                            egui::pos2(
                                point_screen.x / ctx.pixels_per_point() - widget_max_size.x / 2.0,
                                (screen_height as f32 - point_screen.y) / ctx.pixels_per_point(),
                            ),
                            widget_max_size,
                        );
                        ui.put(widget_rect, egui::Label::new(idx.to_string()));
                    }
                };
            });
    }

    fn draw_vis_points(&self, ui: &mut Ui, _scene: &Scene, entity: Entity, command_buffer: &mut CommandBuffer, is_visible: bool, c: &mut VisPoints) {
        // VIS POINTS
        ui.label("Points");
        ui.separator();
        ui.add_enabled_ui(is_visible, |ui| {
            let res = ui.checkbox(&mut c.show_points, "Show points");
            ui.checkbox(&mut c.show_points_indices, "Show point indices");
            if res.clicked() {
                command_buffer.insert_one(entity, ShadowMapDirty);
            }
            //all the other guis are disabled if we don't show points
            if c.show_points {
                //point_color
                ui.horizontal(|ui| {
                    ui.color_edit_button_rgba_premultiplied(&mut c.point_color.data.0.as_mut_slice()[0]);
                    ui.label("Point color");
                });
                //point_size
                ui.horizontal(|ui| {
                    ui.spacing_mut().slider_width = SIDE_PANEL_WIDTH / 3.0; //changes size od slider
                    ui.add(Slider::new(&mut c.point_size, 0.0..=30.0).text("Size"))
                });
                ui.checkbox(&mut c.is_point_size_in_world_space, "isSizeInWorld");
                // zbuffer
                ui.checkbox(&mut c.zbuffer, "Use Z-Buffer");
                //colortype
                egui::ComboBox::new(0, "Color") //the id has to be unique to other comboboxes
                    .selected_text(format!("{:?}", c.color_type))
                    .show_ui(ui, |ui| {
                        // ui.style_mut().wrap = Some(false);
                        ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Wrap);
                        ui.set_min_width(60.0);
                        ui.selectable_value(&mut c.color_type, PointColorType::Solid, "Solid");
                        ui.selectable_value(&mut c.color_type, PointColorType::PerVert, "PerVert");
                    });
            }
        });
    }

    fn draw_vis_mesh(&mut self, ui: &mut Ui, scene: &mut Scene, entity: Entity, command_buffer: &mut CommandBuffer, is_visible: bool) {
        let mut c = scene.get_comp::<&mut VisMesh>(&entity).unwrap();

        // VIS MESH
        ui.label("Mesh");
        ui.separator();

        // Use `add_enabled_ui` to conditionally enable UI elements
        ui.add_enabled_ui(is_visible, |ui| {
            let res = ui.checkbox(&mut c.show_mesh, "Show mesh");
            if res.clicked() {
                command_buffer.insert_one(entity, ShadowMapDirty);
            }

            let _name = scene.get_comp::<&Name>(&entity).unwrap().0.clone();

            // The following settings are only enabled if `show_mesh` is true
            if c.show_mesh {
                // Solid color editor
                ui.horizontal(|ui| {
                    ui.color_edit_button_rgba_unmultiplied(&mut c.solid_color.data.0.as_mut_slice()[0]);
                    ui.label("Solid color");
                });

                // Metalness slider
                ui.horizontal(|ui| {
                    ui.spacing_mut().slider_width = SIDE_PANEL_WIDTH / 3.0;
                    ui.add(Slider::new(&mut c.metalness, 0.0..=1.0).text("Metal"));
                });

                // Roughness slider
                ui.horizontal(|ui| {
                    ui.spacing_mut().slider_width = SIDE_PANEL_WIDTH / 3.0;
                    ui.add(Slider::new(&mut c.perceptual_roughness, 0.0..=1.0).text("Rough"));
                });

                // Roughness black level slider
                ui.horizontal(|ui| {
                    ui.spacing_mut().slider_width = SIDE_PANEL_WIDTH / 3.0;
                    ui.add(Slider::new(&mut c.roughness_black_lvl, 0.0..=1.0).text("RoughBlackLvl"));
                });

                // Opacity slider
                ui.horizontal(|ui| {
                    ui.spacing_mut().slider_width = SIDE_PANEL_WIDTH / 3.0;
                    ui.add(Slider::new(&mut c.opacity, 0.0..=1.0).text("Opacity"));
                });

                // SSS checkbox
                ui.checkbox(&mut c.needs_sss, "Needs SSS");

                // Color type selection combo box
                egui::ComboBox::new(1, "Color")
                    .selected_text(format!("{:?}", c.color_type))
                    .show_ui(ui, |ui| {
                        ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Wrap);
                        ui.set_min_width(60.0);
                        ui.selectable_value(&mut c.color_type, MeshColorType::Solid, "Solid");
                        ui.selectable_value(&mut c.color_type, MeshColorType::PerVert, "PerVert");
                        ui.selectable_value(&mut c.color_type, MeshColorType::Texture, "Texture");
                        ui.selectable_value(&mut c.color_type, MeshColorType::UV, "UV");
                        ui.selectable_value(&mut c.color_type, MeshColorType::Normal, "Normal");
                        ui.selectable_value(&mut c.color_type, MeshColorType::NormalViewCoords, "NormalViewCoords");
                    });

                // UV scale slider
                ui.horizontal(|ui| {
                    ui.spacing_mut().slider_width = SIDE_PANEL_WIDTH / 3.0;
                    ui.add(Slider::new(&mut c.uv_scale, 0.0..=3000.0).text("UVScale"));
                });
            }
        });
    }

    fn draw_vis_lines(
        &mut self,
        ui: &mut Ui,
        _scene: &Scene,
        entity: Entity,
        command_buffer: &mut CommandBuffer,
        is_visible: bool,
        c: &mut VisLines,
    ) {
        ui.label("Lines");
        ui.separator();

        // Use `add_enabled_ui` to conditionally enable UI elements
        ui.add_enabled_ui(is_visible, |ui| {
            let res = ui.checkbox(&mut c.show_lines, "Show lines");
            if res.clicked() {
                command_buffer.insert_one(entity, ShadowMapDirty);
            }

            if c.show_lines {
                // Solid color editor
                ui.horizontal(|ui| {
                    ui.color_edit_button_rgba_unmultiplied(&mut c.line_color.data.0.as_mut_slice()[0]);
                    ui.label("Solid color");
                });

                // Line width slider
                ui.horizontal(|ui| {
                    ui.spacing_mut().slider_width = SIDE_PANEL_WIDTH / 3.0; // Adjust slider width
                    ui.add(Slider::new(&mut c.line_width, 0.0..=30.0).text("Width"));
                });

                // Z-buffer and antialiasing options
                ui.checkbox(&mut c.zbuffer, "Use Z-Buffer");
                ui.checkbox(&mut c.antialias_edges, "Antialias");
            }
        });
    }

    // TODO: Probably remove this, can always add back later if needed
    // fn draw_vis_wireframe(
    //     &mut self,
    //     // ctx: &egui::Context,
    //     ui: &mut Ui,
    //     // renderer: &Renderer,
    //     _scene: &Scene,
    //     entity: Entity,
    //     command_buffer: &mut CommandBuffer,
    //     is_visible: bool,
    //     c: &mut VisWireframe,
    // ) {
    //     //VIS Wireframe
    //     ui.label("Wireframe");
    //     ui.separator();
    //     ui.add_visible_ui(is_visible, |ui| {
    //         let res = ui.checkbox(&mut c.show_wireframe, "Show wireframe");
    //         if res.clicked() {
    //             command_buffer.insert_one(entity, ShadowMapDirty);
    //         }
    //         if c.show_wireframe {
    //             // ui.add_enabled_ui(c.show_wireframe, |ui| {
    //             //solid_color
    //             ui.horizontal(|ui| {
    //                 ui.color_edit_button_rgba_unmultiplied(
    //                     &mut c.wire_color.data.0.as_mut_slice()[0],
    //                 );
    //                 ui.label("Wire color");
    //             });
    //             //width
    //             ui.horizontal(|ui| {
    //                 ui.spacing_mut().slider_width = SIDE_PANEL_WIDTH / 3.0;
    // //changes size od slider                 ui.add(Slider::new(&mut
    // c.wire_width, 0.0..=8.0).text("Width"))             });
    //             // });
    //         }
    //     });
    // }

    // fn draw_vis_normals(
    //     &mut self,
    //     // ctx: &egui::Context,
    //     ui: &mut Ui,
    //     // renderer: &Renderer,
    //     _scene: &Scene,
    //     entity: Entity,
    //     command_buffer: &mut CommandBuffer,
    //     is_visible: bool,
    //     c: &mut VisNormals,
    // ) {
    //     //VIS Normals
    //     ui.label("Normals");
    //     ui.separator();
    //     ui.add_visible_ui(is_visible, |ui| {
    //         let res = ui.checkbox(&mut c.show_normals, "Show Normals");
    //         if res.clicked() {
    //             command_buffer.insert_one(entity, ShadowMapDirty);
    //         }
    //         if c.show_normals {
    //             // ui.add_enabled_ui(c.show_wireframe, |ui| {
    //             //solid_color
    //             ui.horizontal(|ui| {
    //                 ui.color_edit_button_rgba_unmultiplied(
    //                     &mut c.normals_color.data.0.as_mut_slice()[0],
    //                 );
    //                 ui.label("Normal color");
    //             });
    //             //width
    //             ui.horizontal(|ui| {
    //                 ui.spacing_mut().slider_width = SIDE_PANEL_WIDTH / 3.0;
    // //changes size od slider                 ui.add(Slider::new(&mut
    // c.normals_width, 0.0..=8.0).text("Width"))             });
    //             //scale
    //             ui.horizontal(|ui| {
    //                 ui.spacing_mut().slider_width = SIDE_PANEL_WIDTH / 3.0;
    // //changes size od slider                 ui.add(Slider::new(&mut
    // c.normals_scale, 0.0..=1.0).text("Scale"))             });
    //         }
    //     });
    // }

    fn draw_comps(&mut self, ui: &mut Ui, scene: &Scene, entity: Entity, _command_buffer: &mut CommandBuffer, _is_visible: bool) {
        //VIS Normals
        ui.label("Components");
        ui.separator();
        let e_ref = scene.world.entity(entity).unwrap();
        //print component names
        let comp_infos: Vec<gloss_hecs::TypeInfo> = e_ref.component_infos().collect();
        let comp_full_names: Vec<String> = comp_infos.iter().map(gloss_hecs::TypeInfo::name).collect();
        // let mut comp_names: Vec<String> = comp_full_names
        //     .iter()
        //     .map(|n| n.split("::").last().unwrap().to_string())
        //     .collect();
        // Some of our comps are now generic so they need to be handled
        let mut comp_names: Vec<String> = comp_full_names
            .iter()
            .map(|n| {
                // Split at the '<' character if it exists, otherwise split at "::" and take the
                // last part
                if let Some(pos) = n.find('<') {
                    n[..pos].split("::").last().unwrap().to_string()
                } else {
                    n.split("::").last().unwrap().to_string()
                }
            })
            .collect();

        //concat also the type id
        let mut comp_names_and_type = Vec::new();
        for (comp_name, comp_info) in comp_names.iter().zip(comp_infos.iter()) {
            let comp_info_str = format!("{:?}", comp_info.id());
            comp_names_and_type.push(comp_name.to_owned() + &comp_info_str);
        }
        comp_names.sort();
        comp_names_and_type.sort();

        for name in comp_names {
            ui.label(RichText::new(name).font(FontId::proportional(10.0)));
        }
    }

    fn draw_textures(
        &mut self,
        ui: &mut Ui,
        scene: &mut Scene,
        egui_renderer: &mut egui_wgpu::Renderer,
        gpu: &Gpu,
        _command_buffer: &mut CommandBuffer,
    ) {
        if scene.nr_renderables() == 0 {
            return;
        }
        let Some(entity) = scene.get_entity_with_name(&self.selected_mesh_name) else {
            error!("Selected mesh does not exist with name {}", self.selected_mesh_name);
            return;
        };

        //get diffuse tex
        ui.label("Diffuse");
        ui.separator();
        let diffuse_tex = scene.get_comp::<&DiffuseTex>(&entity).unwrap();
        let (view, id) = if scene.world.has::<DiffuseImg>(entity).unwrap() {
            let diffuse_id = diffuse_tex.0.texture.global_id();
            (&diffuse_tex.0.view, diffuse_id)
        } else {
            let diffuse_tex = self.default_texture.as_ref().unwrap();
            let diffuse_id = diffuse_tex.texture.global_id();
            (&diffuse_tex.view, diffuse_id)
        };
        //get egui textureid
        let diffuse_egui_tex_id = self
            .wgputex_2_eguitex
            .entry(id)
            .or_insert_with(|| egui_renderer.register_native_texture(gpu.device(), view, wgpu::FilterMode::Linear));
        //show img
        let res = ui.add(egui::Image::from_texture((*diffuse_egui_tex_id, Vec2::new(120.0, 120.0))));
        self.hovered_diffuse_tex = res.hovered();

        //get normal tex
        ui.label("Normal");
        ui.separator();
        let normal_tex = scene.get_comp::<&NormalTex>(&entity).unwrap();
        let (view, id) = if scene.world.has::<NormalImg>(entity).unwrap() {
            let normal_id = normal_tex.0.texture.global_id();
            (&normal_tex.0.view, normal_id)
        } else {
            let normal_tex = self.default_texture.as_ref().unwrap();
            let normal_id = normal_tex.texture.global_id();
            (&normal_tex.view, normal_id)
        };
        //get egui textureid
        let normal_egui_tex_id = self
            .wgputex_2_eguitex
            .entry(id)
            .or_insert_with(|| egui_renderer.register_native_texture(gpu.device(), view, wgpu::FilterMode::Linear));
        //show img
        let res = ui.add(egui::Image::from_texture((*normal_egui_tex_id, Vec2::new(120.0, 120.0))));
        self.hovered_normal_tex = res.hovered();

        //get normal tex
        ui.label("Roughness");
        ui.separator();
        let roughness_tex = scene.get_comp::<&RoughnessTex>(&entity).unwrap();
        let (view, id) = if scene.world.has::<RoughnessImg>(entity).unwrap() {
            let roughness_id = roughness_tex.0.texture.global_id();
            (&roughness_tex.0.view, roughness_id)
        } else {
            let roughness_tex = self.default_texture.as_ref().unwrap();
            let roughness_id = roughness_tex.texture.global_id();
            (&roughness_tex.view, roughness_id)
        };
        //get egui textureid
        let roughness_egui_tex_id = self
            .wgputex_2_eguitex
            .entry(id)
            .or_insert_with(|| egui_renderer.register_native_texture(gpu.device(), view, wgpu::FilterMode::Linear));
        //show img
        let res = ui.add(egui::Image::from_texture((*roughness_egui_tex_id, Vec2::new(120.0, 120.0))));
        self.hovered_roughness_tex = res.hovered();
    }

    // TODO: Keep or remove?
    // #[allow(clippy::similar_names)]
    // fn draw_move(&mut self, ui: &mut Ui, scene: &Scene, command_buffer: &mut
    // CommandBuffer) {     egui::ComboBox::from_label("Mode")
    //         .selected_text(format!("{:?}", self.gizmo_mode))
    //         .show_ui(ui, |ui| {
    //             ui.selectable_value(&mut self.gizmo_mode, GizmoMode::Rotate,
    // "Rotate");             ui.selectable_value(&mut self.gizmo_mode,
    // GizmoMode::Translate, "Translate");             ui.selectable_value(&mut
    // self.gizmo_mode, GizmoMode::Scale, "Scale");         });

    //     egui::ComboBox::from_label("Orientation")
    //         .selected_text(format!("{:?}", self.gizmo_orientation))
    //         .show_ui(ui, |ui| {
    //             ui.selectable_value(
    //                 &mut self.gizmo_orientation,
    //                 GizmoOrientation::Global,
    //                 "Global",
    //             );
    //             ui.selectable_value(
    //                 &mut self.gizmo_orientation,
    //                 GizmoOrientation::Local,
    //                 "Local",
    //             );
    //         });

    //     //get camera
    //     let cam = scene.get_current_cam().unwrap();
    //     if !cam.is_initialized(scene) {
    //         return;
    //     }
    //     let view = cam.view_matrix(scene);
    //     let proj = cam.proj_matrix(scene);
    //     let v: [[f32; 4]; 4] = view.into();
    //     let p: [[f32; 4]; 4] = proj.into();

    //     //model matrix
    //     // let entity = scene.get_entity_with_name(&self.selected_mesh_name);
    //     if let Some(entity) = self.selected_entity {
    //         let model_matrix = scene.get_comp::<&ModelMatrix>(&entity).unwrap();
    //         let mm: [[f32; 4]; 4] = model_matrix.0.to_homogeneous().into();

    //         let gizmo = Gizmo::new("My gizmo")
    //             .view_matrix(v.into())
    //             .projection_matrix(p.into())
    //             .model_matrix(mm.into())
    //             .mode(self.gizmo_mode)
    //             .orientation(self.gizmo_orientation);

    //         let possible_gizmo_response = gizmo.interact(ui);

    //         if let Some(gizmo_response) = possible_gizmo_response {
    //             let _new_model_matrix = gizmo_response.transform();

    //             //TODO you can actually do gizmo_response.translation and
    // gizmo_reponse.quat directly...

    //             //get T
    //             let new_t_mint = gizmo_response.translation;
    //             let mut new_t = na::Translation3::<f32>::identity();
    //             new_t.x = new_t_mint.x;
    //             new_t.y = new_t_mint.y;
    //             new_t.z = new_t_mint.z;
    //             //get R
    //             let new_q_mint = gizmo_response.rotation;
    //             let mut new_quat = na::Quaternion::<f32>::identity();
    //             new_quat.i = new_q_mint.v.x;
    //             new_quat.j = new_q_mint.v.y;
    //             new_quat.k = new_q_mint.v.z;
    //             new_quat.w = new_q_mint.s;
    //             let new_quat_unit =
    // na::UnitQuaternion::from_quaternion(new_quat);             let new_rot =
    // new_quat_unit.into();             //get scale
    //             let new_scale_mint = gizmo_response.scale;
    //             let new_scale =
    // new_scale_mint.x.max(new_scale_mint.y).max(new_scale_mint.z); //TODO For now
    // we only get one scale value

    //             // let new_scale = gizmo_response.scale.max_element(); //TODO For
    // now we only get one scale value             //
    // // println!("gizmo scale is {:?}", gizmo_response.scale);             //
    // // println!("new_scale {:?}", new_scale);             //combine
    //             let mut new_model_mat = na::SimilarityMatrix3::<f32>::identity();
    //             new_model_mat.append_rotation_mut(&new_rot);
    //             new_model_mat.append_translation_mut(&new_t);
    //             new_model_mat.set_scaling(new_scale);

    //             //set
    //             // mesh.set_model_matrix(scene, new_isometry);
    //             let new_model_matrix = ModelMatrix(new_model_mat);
    //             command_buffer.insert_one(entity, new_model_matrix);

    //             //if it has pos lookat we also modify that
    //             if scene.world.has::<PosLookat>(entity).unwrap() {
    //                 let pos_lookat =
    // scene.get_comp::<&PosLookat>(&entity).unwrap();                 // TODO
    // make poslookat clone                 let pos_lookat =
    //                     PosLookat::new_from_model_matrix(new_model_mat,
    // pos_lookat.dist_lookat());
    // command_buffer.insert_one(entity, pos_lookat);             }
    //         }
    //     }
    // }

    fn draw_params(&mut self, ui: &mut Ui, _scene: &mut Scene, config: &mut Config, _command_buffer: &mut CommandBuffer) {
        //TODO get all things from config

        //ambient factor
        ui.horizontal(|ui| {
            ui.spacing_mut().slider_width = SIDE_PANEL_WIDTH / 3.0; //changes size od slider
            ui.add(Slider::new(&mut config.render.ambient_factor, 0.0..=1.0).text("AmbientFactor"))
        });
        //environment_factor
        ui.horizontal(|ui| {
            ui.spacing_mut().slider_width = SIDE_PANEL_WIDTH / 3.0; //changes size od slider
            ui.add(Slider::new(&mut config.render.environment_factor, 0.0..=5.0).text("EnvFactor"))
        });
        //bg_color
        ui.horizontal(|ui| {
            ui.color_edit_button_rgba_premultiplied(&mut config.render.bg_color.data.0.as_mut_slice()[0]);
            ui.label("bg_color");
        });
        //distance fade
        ui.checkbox(config.render.enable_distance_fade.as_mut().unwrap_or(&mut false), "DistanceFade");
        //distance fade start
        ui.horizontal(|ui| {
            ui.spacing_mut().slider_width = SIDE_PANEL_WIDTH / 3.0; //changes size od slider
            ui.add(Slider::new(config.render.distance_fade_start.as_mut().unwrap_or(&mut 0.0), 1.0..=100.0).text("FadeStart"))
        });
        //distance fade end
        ui.horizontal(|ui| {
            ui.spacing_mut().slider_width = SIDE_PANEL_WIDTH / 3.0; //changes size od slider
            ui.add(Slider::new(config.render.distance_fade_end.as_mut().unwrap_or(&mut 0.0), 1.0..=100.0).text("FadeEnd"))
        });
        //saturation
        ui.horizontal(|ui| {
            ui.spacing_mut().slider_width = SIDE_PANEL_WIDTH / 3.0; //changes size od slider
            ui.add(Slider::new(&mut config.render.saturation, 0.0..=2.0).text("Saturation"))
        });
        //gamma
        ui.horizontal(|ui| {
            ui.spacing_mut().slider_width = SIDE_PANEL_WIDTH / 3.0; //changes size od slider
            ui.add(Slider::new(&mut config.render.gamma, 0.5..=1.5).text("Gamma"))
        });
        //exposure
        ui.horizontal(|ui| {
            ui.spacing_mut().slider_width = SIDE_PANEL_WIDTH / 3.0; //changes size od slider
            ui.add(Slider::new(&mut config.render.exposure, -5.0..=5.0).text("Exposure"))
        });
    }

    #[allow(clippy::too_many_lines)]
    fn draw_lights(&mut self, ui: &mut Ui, scene: &mut Scene, command_buffer: &mut CommandBuffer) {
        // //get all entities that are renderable and sort by name
        let entities = scene.get_lights(true);

        //go through all the lights and show their name
        ui.group(|ui| {
            ScrollArea::vertical()
                .max_height(200.0)
                .scroll_bar_visibility(scroll_area::ScrollBarVisibility::AlwaysVisible)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.with_layout(Layout::top_down_justified(Align::LEFT), |ui| {
                        ui.spacing_mut().item_spacing.y = 0.0;
                        ui.spacing_mut().button_padding.y = 4.0;
                        for entity in entities {
                            let e_ref = scene.world.entity(entity).unwrap();

                            //get the name of the mesh which acts like a unique id
                            let name = e_ref.get::<&Name>().expect("The entity has no name").0.clone();

                            //if it's the first time we encounter a renderable mesh,  we set the selected
                            // name to this one
                            if self.selected_light_name.is_empty() {
                                self.selected_light_name.clone_from(&name);
                            }

                            //GUI for this concrete mesh
                            //if we click we can see options for vis
                            let _res = ui.selectable_value(&mut self.selected_light_name, name.clone(), &name);

                            if name == self.selected_light_name {
                                self.selected_light_entity = Some(entity);
                            }
                        }
                    });
                });
        });

        if let Some(entity) = self.selected_light_entity {
            ui.label("LightEmit");
            ui.separator();
            //color
            ui.horizontal(|ui| {
                let mut comp_light_emit = scene.get_comp::<&mut LightEmit>(&entity).unwrap();
                ui.color_edit_button_rgb(&mut comp_light_emit.color.data.0.as_mut_slice()[0]);
                ui.label("color");
            });
            //intensity
            ui.horizontal(|ui| {
                let mut comp_light_emit = scene.get_comp::<&mut LightEmit>(&entity).unwrap();
                ui.spacing_mut().slider_width = SIDE_PANEL_WIDTH / 3.0; //changes size od slider
                ui.add(Slider::new(&mut comp_light_emit.intensity, 0.0..=1_000_000.0).text("intensity"))
            });
            //cast shadows
            let mut is_shadow_casting: bool = scene.world.has::<ShadowCaster>(entity).unwrap();
            let res = ui.checkbox(&mut is_shadow_casting, "CastShadows");
            if res.changed() {
                if is_shadow_casting {
                    //TODO adding this parameter just adds it with these daault. Maybe we need
                    // another way to temporarily disable shadows
                    command_buffer.insert_one(
                        entity,
                        ShadowCaster {
                            shadow_res: 2048,
                            shadow_bias_fixed: 2e-5,
                            shadow_bias: 0.15,
                            shadow_bias_normal: 1.5,
                        },
                    );
                } else {
                    command_buffer.remove_one::<ShadowCaster>(entity);
                }
            }
            // shadow_res
            if scene.world.has::<ShadowCaster>(entity).unwrap() {
                //we only mutate the component if we modify the slider because we want the
                // change detection of hecs to only pick-up and detect when we actually change
                // the value
                let shadow_caster_read = scene.get_comp::<&ShadowCaster>(&entity).unwrap();
                let mut shadow_res = shadow_caster_read.shadow_res;
                let mut shadow_bias_fixed = shadow_caster_read.shadow_bias_fixed;
                let mut shadow_bias = shadow_caster_read.shadow_bias;
                let mut shadow_bias_normal = shadow_caster_read.shadow_bias_normal;
                drop(shadow_caster_read);
                // ui.horizontal(|ui| {
                ui.spacing_mut().slider_width = SIDE_PANEL_WIDTH / 3.0; //changes size od slider
                let res = ui.add(Slider::new(&mut shadow_res, 128..=4096).text("shadow_res"));
                if res.changed() {
                    //now we actually mutate the component
                    let mut shadow_caster = scene.get_comp::<&mut ShadowCaster>(&entity).unwrap();
                    shadow_caster.shadow_res = shadow_res;
                }
                //shadow bias fixed
                let res = ui.add(Slider::new(&mut shadow_bias_fixed, 0.0..=0.5).text("shadow_bias_fixed"));
                if res.changed() {
                    //now we actually mutate the component
                    let mut shadow_caster = scene.get_comp::<&mut ShadowCaster>(&entity).unwrap();
                    shadow_caster.shadow_bias_fixed = shadow_bias_fixed;
                }

                //shadow bias light
                let res = ui.add(Slider::new(&mut shadow_bias, 0.0..=0.4).text("shadow_bias"));
                if res.changed() {
                    //now we actually mutate the component
                    let mut shadow_caster = scene.get_comp::<&mut ShadowCaster>(&entity).unwrap();
                    shadow_caster.shadow_bias = shadow_bias;
                }
                //shadow bias normal
                let res = ui.add(Slider::new(&mut shadow_bias_normal, 0.0..=50.0).text("shadow_bias_normal"));
                if res.changed() {
                    //now we actually mutate the component
                    let mut shadow_caster = scene.get_comp::<&mut ShadowCaster>(&entity).unwrap();
                    shadow_caster.shadow_bias_normal = shadow_bias_normal;
                }
                // });
            }
            //range
            ui.horizontal(|ui| {
                let mut comp_light_emit = scene.get_comp::<&mut LightEmit>(&entity).unwrap();
                ui.spacing_mut().slider_width = SIDE_PANEL_WIDTH / 3.0; //changes size od slider
                ui.add(Slider::new(&mut comp_light_emit.range, 0.0..=10000.0).text("range"))
            });
            //radius
            ui.horizontal(|ui| {
                let mut comp_light_emit = scene.get_comp::<&mut LightEmit>(&entity).unwrap();
                ui.spacing_mut().slider_width = SIDE_PANEL_WIDTH / 3.0; //changes size od slider
                ui.add(Slider::new(&mut comp_light_emit.radius, 0.0..=10.0).text("radius"))
            });
            //inner_angle
            ui.horizontal(|ui| {
                let mut comp_light_emit = scene.get_comp::<&mut LightEmit>(&entity).unwrap();
                let outer_angle = comp_light_emit.outer_angle;
                ui.spacing_mut().slider_width = SIDE_PANEL_WIDTH / 3.0; //changes size od slider
                ui.add(Slider::new(&mut comp_light_emit.inner_angle, 0.0..=outer_angle).text("inner_angle"))
            });
            //outer_angle
            ui.horizontal(|ui| {
                let mut comp_light_emit = scene.get_comp::<&mut LightEmit>(&entity).unwrap();
                ui.spacing_mut().slider_width = SIDE_PANEL_WIDTH / 3.0; //changes size od slider
                ui.add(Slider::new(&mut comp_light_emit.outer_angle, 0.0..=std::f32::consts::PI / 2.0).text("outer_angle"))
            });
            ui.horizontal(|ui| {
                let mut comp_proj = scene.get_comp::<&mut Projection>(&entity).unwrap();
                let (mut near, _) = comp_proj.near_far();
                ui.spacing_mut().slider_width = SIDE_PANEL_WIDTH / 3.0; //changes size od slider
                let res = ui.add(Slider::new(&mut near, 0.0..=20.0).text("near"));
                if res.changed() {
                    comp_proj.set_near(near);
                }
            });
            //far
            ui.horizontal(|ui| {
                let mut comp_proj = scene.get_comp::<&mut Projection>(&entity).unwrap();
                let (_, mut far) = comp_proj.near_far();
                ui.spacing_mut().slider_width = SIDE_PANEL_WIDTH / 3.0; //changes size od slider
                let res = ui.add(Slider::new(&mut far, 0.0..=2000.0).text("far"));
                if res.changed() {
                    comp_proj.set_far(far);
                }
            });
            {
                let mut poslookat = scene.get_comp::<&mut PosLookat>(&entity).unwrap();
                ui.spacing_mut().slider_width = SIDE_PANEL_WIDTH / 3.0; //changes size od slider
                ui.add(Slider::new(&mut poslookat.position.x, -40.0..=40.0).text("x"));
                ui.add(Slider::new(&mut poslookat.position.y, -40.0..=40.0).text("y"));
                ui.add(Slider::new(&mut poslookat.position.z, -40.0..=40.0).text("z"));
            }
            //view plane that cna be used to manipulate the light
            let mut has_mesh = scene.world.has::<Renderable>(entity).unwrap();
            let res = ui.checkbox(&mut has_mesh, "ShowLight");
            if res.changed() {
                if has_mesh {
                    //add plane
                    let center = scene.get_comp::<&PosLookat>(&entity).unwrap().position;
                    let normal = scene.get_comp::<&PosLookat>(&entity).unwrap().direction();
                    //move the plane a bit behind the light so it doesn't cast a shadow
                    let center = center - normal * 0.03;
                    let mut builder = Geom::build_plane(center, normal, 0.3, 0.3, false);
                    let _ = scene.world.insert(entity, builder.build());
                    let _ = scene.world.insert_one(
                        entity,
                        VisMesh {
                            solid_color: na::Vector4::<f32>::new(1.0, 1.0, 1.0, 1.0),
                            ..Default::default()
                        },
                    );
                    let _ = scene.world.insert_one(entity, Renderable);
                } else {
                    command_buffer.remove_one::<Renderable>(entity);
                }
            }
        }
    }

    #[allow(clippy::too_many_lines)]
    fn draw_cam(&mut self, ui: &mut Ui, scene: &mut Scene, _command_buffer: &mut CommandBuffer) {
        // get all entities that are renderable and sort by name
        let _entities = scene.get_lights(true);
        let cam = scene.get_current_cam().unwrap();

        if let Ok(mut projection) = scene.world.get::<&mut Projection>(cam.entity) {
            let (mut near, mut far) = projection.near_far();
            ui.label("Projection");
            ui.separator();
            ui.spacing_mut().slider_width = SIDE_PANEL_WIDTH / 3.0; //changes size od slider
            let res = ui.add(Slider::new(&mut near, 1e-5..=1.0).text("near"));
            if res.changed() {
                projection.set_near(near);
            }
            let res = ui.add(Slider::new(&mut far, near..=10000.0).text("far"));
            if res.changed() {
                projection.set_far(far);
            }
        }
    }

    #[allow(clippy::too_many_lines)]
    #[allow(clippy::cast_precision_loss)]
    fn draw_plugins(&mut self, ui: &mut Ui, _scene: &mut Scene, plugins: &Plugins, _command_buffer: &mut CommandBuffer) {
        // //get all entities that are renderable and sort by name
        // let entities = scene.get_lights(true);
        egui::Grid::new("grid_plugins").show(ui, |ui| {
            for system_and_metadata in plugins.logic_systems.iter() {
                let metadata = &system_and_metadata.1;
                let sys = &system_and_metadata.0;
                let name = sys.name.clone().unwrap_or(RString::from("unknown_name"));
                let ms = metadata.execution_time.as_nanos() as f32 / 1_000_000.0;
                let ms_string = format!("{ms:.2}");
                //draw in green if the system took more than 0.01ms (todo we need a better way
                // to check if the system was ran)
                let was_ran = ms > 0.07;
                // ui.with_layout(egui::Layout::left_to_right(egui::Align::TOP), |ui| {
                if was_ran {
                    ui.label(egui::RichText::new(name).strong());
                    ui.label(egui::RichText::new(ms_string).strong());
                } else {
                    ui.label(egui::RichText::new(name).weak());
                    ui.label(egui::RichText::new(ms_string).weak());
                }
                ui.end_row();
            }
        });

        // //go through all the lights and show their name
        // TODO: Keep or remove?
        // ui.group(|ui| {
        //     ScrollArea::vertical()
        //         .max_height(200.0)
        //         .scroll_bar_visibility(scroll_area::ScrollBarVisibility::AlwaysVisible)
        //         .auto_shrink([false, false])
        //         .show(ui, |ui| {
        //             ui.with_layout(Layout::top_down_justified(Align::LEFT),
        // |ui| {                 ui.spacing_mut().item_spacing.y = 0.0;
        //                 ui.spacing_mut().button_padding.y = 4.0;
        //                 for entity in entities {
        //                     let e_ref = scene.world.entity(entity).unwrap();

        //                     //get the name of the mesh which acts like a
        // unique id                     let name = e_ref
        //                         .get::<&Name>()
        //                         .expect("The entity has no name")
        //                         .0
        //                         .clone();

        //                     //if it's the first time we encounter a
        // renderable mesh,  we set the selected name to this one
        //                     if self.selected_light_name.is_empty() {
        //                         self.selected_light_name = name.clone();
        //                     }

        //                     //GUI for this concrete mesh
        //                     //if we click we can see options for vis
        //                     let _res = ui.selectable_value(
        //                         &mut self.selected_light_name,
        //                         name.clone(),
        //                         &name,
        //                     );

        //                     if name == self.selected_light_name {
        //                         self.selected_light_entity = Some(entity);
        //                     }
        //                 }
        //             });
        //         });
        // });
    }

    #[allow(clippy::too_many_lines)]
    fn draw_io(&mut self, ui: &mut Ui, scene: &mut Scene, _command_buffer: &mut CommandBuffer, selected_entity: Option<Entity>) {
        //save obj
        if let Some(selected_entity) = selected_entity {
            if ui.add(egui::Button::new("Save Obj")).clicked() {
                //get v, f and possibly uv from the entity
                if scene.world.has::<Verts>(selected_entity).unwrap() && scene.world.has::<Faces>(selected_entity).unwrap() {
                    let v = scene.get_comp::<&Verts>(&selected_entity).unwrap();
                    // let uv = scene.get_comp::<&UVs>(&selected_entity);

                    //transform vertices
                    let mm = scene.get_comp::<&ModelMatrix>(&selected_entity).unwrap();
                    let v = Geom::transform_verts(&v.0.to_dmatrix(), &mm.0);

                    let f = scene.get_comp::<&Faces>(&selected_entity).ok().map(|f| f.0.clone());

                    let uv = scene.get_comp::<&UVs>(&selected_entity).ok().map(|uv| uv.0.clone());

                    let normals = scene
                        .get_comp::<&Normals>(&selected_entity)
                        .ok()
                        .map(|normals| Geom::transform_vectors(&normals.0.to_dmatrix(), &mm.0));

                    //TODO make the path parametrizable
                    Geom::save_obj(
                        &v,
                        f.map(|faces| faces.to_dmatrix()).as_ref(),
                        // None,
                        uv.map(|faces| faces.to_dmatrix()).as_ref(),
                        normals.as_ref(),
                        "./saved_obj.obj",
                    );
                }
            }
        }

        //save ply
        if let Some(selected_entity) = selected_entity {
            if ui.add(egui::Button::new("Save Ply")).clicked() {
                //get v, f and possibly uv from the entity
                if scene.world.has::<Verts>(selected_entity).unwrap() && scene.world.has::<Faces>(selected_entity).unwrap() {
                    let v = scene.get_comp::<&Verts>(&selected_entity).unwrap();

                    //transform vertices
                    let mm = scene.get_comp::<&ModelMatrix>(&selected_entity).unwrap();
                    let v = Geom::transform_verts(&v.0.to_dmatrix(), &mm.0);

                    let f = scene.get_comp::<&Faces>(&selected_entity).ok().map(|f| f.0.clone());

                    let uv = scene.get_comp::<&UVs>(&selected_entity).ok().map(|uv| uv.0.clone());

                    let normals = scene
                        .get_comp::<&Normals>(&selected_entity)
                        .ok()
                        .map(|normals| Geom::transform_vectors(&normals.0.to_dmatrix(), &mm.0));

                    let colors = scene.get_comp::<&Colors>(&selected_entity).ok().map(|colors| colors.0.clone());

                    //TODO make the path parametrizable
                    Geom::save_ply(
                        &v,
                        f.map(|faces| faces.to_dmatrix()).as_ref(),
                        // None,
                        uv.map(|uvs| uvs.to_dmatrix()).as_ref(),
                        normals.as_ref(),
                        colors.map(|colors| colors.to_dmatrix()).as_ref(),
                        "./saved_ply.ply",
                    );
                }
            }
        }
    }

    #[allow(clippy::too_many_lines)]
    #[allow(unused_variables)]
    fn draw_profiling(&mut self, ui: &mut Ui, _scene: &mut Scene, _command_buffer: &mut CommandBuffer) {
        // TODO: Keep or remove?
        // cfg_if::cfg_if! {
        //     if #[cfg(feature = "peak-alloc")] {
        //         let current_mem = PEAK_ALLOC.current_usage_as_mb() as u32;
        //         let peak_mem = PEAK_ALLOC.peak_usage_as_mb() as u32;
        //         ui.label("current_mem (MB): ".to_owned() + &current_mem.to_string());
        //         ui.label("peak_mem(MB): ".to_owned() + &peak_mem.to_string());
        //     }
        // }

        // cfg_if::cfg_if! {
        //     if #[cfg(feature = "talc")] {

        //         cfg_if::cfg_if! {
        //         if #[cfg(target_arch = "wasm32")] {
        //             use crate::ALLOCATOR;
        //             let talc = ALLOCATOR.lock();
        //             let counters = talc.get_counters();
        //             ui.label("available to claim MB: ".to_owned() +
        // &(counters.available_bytes/(1024*1024)).to_string());
        // ui.label("nr gaps: ".to_owned() + &counters.fragment_count.to_string());
        //             ui.label("claimed MB: ".to_owned() +
        // &(counters.claimed_bytes/(1024*1024)).to_string());
        // ui.label("unavailable MB: ".to_owned() +
        // &(counters.overhead_bytes()/(1024*1024)).to_string());
        // ui.label("total_freed MB: ".to_owned() +
        // &(counters.total_freed_bytes()/(1024*1024)).to_string());         }}
        //     }
        // }

        // cfg_if::cfg_if! {
        //     if #[cfg(feature = "jemallocator")] {

        //         ui.label("current_mem (MB): ".to_owned());
        //     }
        // }

        // cfg_if::cfg_if! {
        //     if #[cfg(feature = "mimalloc")] {
        //         ui.label("mimalloc");

        //         ui.label("current_mem (MB): ".to_owned());
        //     }
        // }

        //for accounting allocator
        let memory_usage = MemoryUse::capture();
        if let Some(mem_resident) = memory_usage.resident {
            ui.label(egui::RichText::new(
                "MB resident: ".to_owned() + &(mem_resident / (1024 * 1024)).to_string(),
            ));
        }
        if let Some(mem_counted) = memory_usage.counted {
            ui.label(egui::RichText::new(
                "MB counted: ".to_owned() + &(mem_counted / (1024 * 1024)).to_string(),
            ));
        }

        //are we tracking callstacks
        let mut is_tracking = gloss_memory::accounting_allocator::is_tracking_callstacks();
        let res = ui.checkbox(&mut is_tracking, "Track memory callstacks");
        if res.clicked() {
            gloss_memory::accounting_allocator::set_tracking_callstacks(is_tracking);
        }

        //make a memory bar where we show all the memory and the parts that are free in
        // red, parts that are allocated in red only works in wasm because it
        // uses a linear memory model
        let size_per_mb = 1.0;
        #[cfg(target_arch = "wasm32")]
        {
            use egui::{epaint::RectShape, Rect, Sense, Shape};
            let unknown_mem = Color32::from_rgb(150, 150, 150);
            let allocated_mem = Color32::from_rgb(255, 10, 10);

            //make per mb colors
            let mut mb2color: Vec<egui::Color32> = Vec::new();
            if let Some(mem_resident) = memory_usage.resident {
                let mb_resident = mem_resident / (1024 * 1024);
                mb2color.resize(mb_resident, unknown_mem);
                let live_allocs = accounting_allocator::live_allocs_list();
                let min_ptr = accounting_allocator::min_ptr_alloc_memory();
                // println!("live_allocs {}", live_allocs.len());
                //for each live allow, paint the value with Red( allocated )
                for (ptr, size) in live_allocs {
                    let ptr_mb = (ptr - min_ptr) / (1024 * 1024);
                    let size_mb = size / (1024 * 1024);
                    //for each mb paint it allocated
                    for local_mb_idx in 0..size_mb {
                        let idx = ptr_mb + local_mb_idx;
                        // println!(
                        //     " ptr_mb {}  ptr{} min_ptr{} local_mb_idx {} idx {}",
                        //     ptr_mb, ptr, min_ptr, local_mb_idx, idx
                        // );
                        if idx < mb2color.len() {
                            mb2color[idx] = allocated_mem;
                        }
                    }
                }
            }

            //draw bar
            if let Some(mem_resident) = memory_usage.resident {
                let mb_resident = mem_resident / (1024 * 1024);
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(0.5, 0.0);
                    // for i in 0..mb_resident {
                    for i in 0..mb_resident {
                        // let cursor = Ui::cursor(ui);
                        let rect_size = egui::Vec2::new(size_per_mb, 10.0);
                        let rect = ui.allocate_exact_size(rect_size, Sense::click()).0;
                        let rounding = Rounding::default();
                        #[allow(arithmetic_overflow)]
                        // let fill_color = egui::Color32::from_rgb(50 * 3, 0, 0);
                        // let fill_color =
                        let fill_color = if i < mb2color.len() {
                            mb2color[i]
                        } else {
                            egui::Color32::from_rgb(50, 0, 0)
                        };
                        let stroke = Stroke::default();
                        let rect_shape = RectShape::new(rect, rounding, fill_color, stroke);
                        ui.painter().add(Shape::Rect(rect_shape));
                    }
                });
            }
        }

        ui.add_space(15.0);
        if let Some(tracks) = accounting_allocator::tracking_stats() {
            #[allow(clippy::cast_precision_loss)]
            for (i, cb) in tracks.top_callstacks.iter().enumerate() {
                //callstack name will be the nr mb
                let mb_total = cb.extant.size as f32 / (1024.0 * 1024.0);
                let text_header =
                    egui::RichText::new(float2string(mb_total, 1) + " MB: " + &cb.readable_backtrace.last_relevant_func_name).size(12.0);

                ui.push_id(i, |ui| {
                    egui::CollapsingHeader::new(text_header).show(ui, |ui| self.draw_callstack_profiling(ui, cb));
                });
            }
        }
    }

    #[allow(clippy::too_many_lines)]
    #[allow(unused_variables)]
    fn draw_callstack_profiling(&mut self, ui: &mut Ui, cb: &CallstackStatistics) {
        let text = "cb: ".to_owned() + &cb.readable_backtrace.to_string();
        ui.label(egui::RichText::new(text).size(11.0));
        ui.label("count".to_owned() + &(cb.extant.count).to_string());
        ui.label("size".to_owned() + &(cb.extant.size / (1024 * 1024)).to_string());
        ui.label("stochastic_rate".to_owned() + &(cb.stochastic_rate).to_string());
    }
}

// Generated by egui-themer (https://github.com/grantshandy/egui-themer).
#[allow(clippy::too_many_lines)]
pub fn style() -> Style {
    Style {
        spacing: Spacing {
            item_spacing: Vec2 { x: 8.0, y: 3.0 },
            window_margin: Margin {
                left: 6.0,
                right: 6.0,
                top: 6.0,
                bottom: 6.0,
            },
            button_padding: Vec2 { x: 4.0, y: 1.0 },
            menu_margin: Margin {
                left: 6.0,
                right: 6.0,
                top: 6.0,
                bottom: 6.0,
            },
            indent: 18.0,
            interact_size: Vec2 { x: 40.0, y: 18.0 },
            slider_width: 100.0,
            combo_width: 100.0,
            text_edit_width: 280.0,
            icon_width: 14.0,
            icon_width_inner: 8.0,
            icon_spacing: 4.0,
            tooltip_width: 600.0,
            indent_ends_with_horizontal_line: false,
            combo_height: 200.0,
            scroll: ScrollStyle {
                bar_width: 6.0,
                handle_min_length: 12.0,
                bar_inner_margin: 4.0,
                bar_outer_margin: 0.0,
                ..Default::default()
            },
            ..Default::default()
        },
        interaction: Interaction {
            resize_grab_radius_side: 5.0,
            resize_grab_radius_corner: 10.0,
            show_tooltips_only_when_still: true,
            tooltip_delay: 0.5,
            ..Default::default()
        },
        visuals: Visuals {
            dark_mode: true,
            override_text_color: None,
            widgets: Widgets {
                noninteractive: WidgetVisuals {
                    bg_fill: Color32::from_rgba_premultiplied(27, 27, 27, 255),
                    weak_bg_fill: Color32::from_rgba_premultiplied(27, 27, 27, 255),
                    bg_stroke: Stroke {
                        width: 1.0,
                        color: Color32::from_rgba_premultiplied(60, 60, 60, 255),
                    },
                    rounding: Rounding {
                        nw: 0.0,
                        ne: 0.0,
                        sw: 0.0,
                        se: 0.0,
                    },
                    fg_stroke: Stroke {
                        width: 1.0,
                        color: Color32::from_rgba_premultiplied(140, 140, 140, 255),
                    },
                    expansion: 0.0,
                },
                inactive: WidgetVisuals {
                    bg_fill: Color32::from_rgba_premultiplied(60, 60, 60, 255),
                    weak_bg_fill: Color32::from_rgba_premultiplied(60, 60, 60, 255),
                    bg_stroke: Stroke {
                        width: 0.0,
                        color: Color32::from_rgba_premultiplied(0, 0, 0, 0),
                    },
                    rounding: Rounding {
                        nw: 0.0,
                        ne: 0.0,
                        sw: 0.0,
                        se: 0.0,
                    },
                    fg_stroke: Stroke {
                        width: 1.0,
                        color: Color32::from_rgba_premultiplied(180, 180, 180, 255),
                    },
                    expansion: 0.0,
                },
                hovered: WidgetVisuals {
                    bg_fill: Color32::from_rgba_premultiplied(70, 70, 70, 255),
                    weak_bg_fill: Color32::from_rgba_premultiplied(70, 70, 70, 255),
                    bg_stroke: Stroke {
                        width: 1.0,
                        color: Color32::from_rgba_premultiplied(150, 150, 150, 255),
                    },
                    rounding: Rounding {
                        nw: 0.0,
                        ne: 0.0,
                        sw: 0.0,
                        se: 0.0,
                    },
                    fg_stroke: Stroke {
                        width: 1.5,
                        color: Color32::from_rgba_premultiplied(240, 240, 240, 255),
                    },
                    expansion: 1.0,
                },
                active: WidgetVisuals {
                    bg_fill: Color32::from_rgba_premultiplied(55, 55, 55, 255),
                    weak_bg_fill: Color32::from_rgba_premultiplied(55, 55, 55, 255),
                    bg_stroke: Stroke {
                        width: 1.0,
                        color: Color32::from_rgba_premultiplied(255, 255, 255, 255),
                    },
                    rounding: Rounding {
                        nw: 0.0,
                        ne: 0.0,
                        sw: 0.0,
                        se: 0.0,
                    },
                    fg_stroke: Stroke {
                        width: 2.0,
                        color: Color32::from_rgba_premultiplied(255, 255, 255, 255),
                    },
                    expansion: 1.0,
                },
                open: WidgetVisuals {
                    bg_fill: Color32::from_rgba_premultiplied(27, 27, 27, 255),
                    weak_bg_fill: Color32::from_rgba_premultiplied(27, 27, 27, 255),
                    bg_stroke: Stroke {
                        width: 1.0,
                        color: Color32::from_rgba_premultiplied(60, 60, 60, 255),
                    },
                    rounding: Rounding {
                        nw: 0.0,
                        ne: 0.0,
                        sw: 0.0,
                        se: 0.0,
                    },
                    fg_stroke: Stroke {
                        width: 1.0,
                        color: Color32::from_rgba_premultiplied(210, 210, 210, 255),
                    },
                    expansion: 0.0,
                },
            },
            selection: Selection {
                bg_fill: Color32::from_rgba_premultiplied(0, 92, 128, 255),
                stroke: Stroke {
                    width: 1.0,
                    color: Color32::from_rgba_premultiplied(192, 222, 255, 255),
                },
            },
            hyperlink_color: Color32::from_rgba_premultiplied(90, 170, 255, 255),
            faint_bg_color: Color32::from_rgba_premultiplied(0, 0, 0, 0),
            extreme_bg_color: Color32::from_rgba_premultiplied(17, 17, 17, 255),
            code_bg_color: Color32::from_rgba_premultiplied(64, 64, 64, 255),
            warn_fg_color: Color32::from_rgba_premultiplied(255, 143, 0, 255),
            error_fg_color: Color32::from_rgba_premultiplied(255, 0, 0, 255),
            window_rounding: Rounding {
                nw: 0.0,
                ne: 0.0,
                sw: 0.0,
                se: 0.0,
            },
            window_shadow: Shadow {
                // extrusion: 5.0,
                color: Color32::from_rgba_premultiplied(0, 0, 0, 96),
                ..Default::default()
            },
            window_fill: Color32::from_rgba_premultiplied(27, 27, 27, 255),
            window_stroke: Stroke {
                width: 0.0,
                color: Color32::from_rgba_premultiplied(71, 71, 71, 255),
            },
            menu_rounding: Rounding {
                nw: 6.0,
                ne: 6.0,
                sw: 6.0,
                se: 6.0,
            },
            panel_fill: Color32::from_rgba_premultiplied(20, 20, 20, 255),
            popup_shadow: Shadow {
                // extrusion: 16.0,
                color: Color32::from_rgba_premultiplied(0, 0, 0, 96),
                ..Default::default()
            },
            resize_corner_size: 12.0,
            // text_cursor_width: 2.0,
            text_cursor: TextCursorStyle::default(),
            // text_cursor_preview: false,
            clip_rect_margin: 3.0,
            button_frame: true,
            collapsing_header_frame: false,
            indent_has_left_vline: true,
            striped: false,
            slider_trailing_fill: true,
            window_highlight_topmost: true,
            handle_shape: HandleShape::Circle,
            interact_cursor: None,
            image_loading_spinners: true,
            numeric_color_space: NumericColorSpace::GammaByte,
        },
        animation_time: 0.08333,
        explanation_tooltips: false,
        ..Default::default()
    }
}
