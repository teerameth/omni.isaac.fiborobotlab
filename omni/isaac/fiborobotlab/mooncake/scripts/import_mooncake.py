# Copyright (c) 2018-2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from torch import roll
import omni
import omni.kit.commands
import asyncio
import math
import weakref
import omni.ui as ui
from omni.kit.menu.utils import add_menu_items, remove_menu_items, MenuItemDescription

from .common import set_drive_parameters
from pxr import UsdLux, Sdf, Gf, UsdPhysics

from omni.isaac.ui.ui_utils import setup_ui_headers, get_style, btn_builder

EXTENSION_NAME = "Import Mooncake"

def velocity2omega(v_x, v_y, w_z=0, d=0.105, r=0.1):
    omega_0 = (v_x-d*w_z)/r
    omega_1 = -(v_x-math.sqrt(3)*v_y+2*d*w_z)/(2*r)
    omega_2 = -(v_x+math.sqrt(3)*v_y+2*d*w_z)/(2*r)
    return [omega_0, omega_1, omega_2]

class Extension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        ext_manager = omni.kit.app.get_app().get_extension_manager()
        self._ext_id = ext_id
        self._extension_path = ext_manager.get_extension_path(ext_id)

        self._menu_items = [
            MenuItemDescription(
                name="Import Robots",
                sub_menu=[
                    MenuItemDescription(name="Mooncake URDF", onclick_fn=lambda a=weakref.proxy(self): a._menu_callback())
                ],
            )
        ]
        add_menu_items(self._menu_items, "Isaac Examples")

        self._build_ui()

    def _build_ui(self):
        self._window = omni.ui.Window(
            EXTENSION_NAME, width=0, height=0, visible=False, dockPreference=ui.DockPreference.LEFT_BOTTOM
        )
        with self._window.frame:
            with ui.VStack(spacing=5, height=0):

                title = "Import a Mooncake Robot via URDF"
                doc_link = "https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/ext_omni_isaac_urdf.html"

                overview = "This Example shows you import an NVIDIA Mooncake robot via URDF.\n\nPress the 'Open in IDE' button to view the source code."

                setup_ui_headers(self._ext_id, __file__, title, doc_link, overview)

                frame = ui.CollapsableFrame(
                    title="Command Panel",
                    height=0,
                    collapsed=False,
                    style=get_style(),
                    style_type_name_override="CollapsableFrame",
                    horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
                )
                with frame:
                    with ui.VStack(style=get_style(), spacing=5):
                        dict = {
                            "label": "Load Robot",
                            "type": "button",
                            "text": "Load",
                            "tooltip": "Load a UR10 Robot into the Scene",
                            "on_clicked_fn": self._on_load_robot,
                        }
                        btn_builder(**dict)

                        dict = {
                            "label": "Configure Drives",
                            "type": "button",
                            "text": "Configure",
                            "tooltip": "Configure Joint Drives",
                            "on_clicked_fn": self._on_config_robot,
                        }
                        btn_builder(**dict)

                        dict = {
                            "label": "Spin Robot",
                            "type": "button",
                            "text": "move",
                            "tooltip": "Spin the Robot in Place",
                            "on_clicked_fn": self._on_config_drives,
                        }
                        btn_builder(**dict)

    def on_shutdown(self):
        remove_menu_items(self._menu_items, "Isaac Examples")
        self._window = None

    def _menu_callback(self):
        self._window.visible = not self._window.visible

    def _on_load_robot(self):
        load_stage = asyncio.ensure_future(omni.usd.get_context().new_stage_async())
        asyncio.ensure_future(self._load_mooncake(load_stage))

    async def _load_mooncake(self, task):
        done, pending = await asyncio.wait({task})
        if task in done:
            status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
            import_config.merge_fixed_joints = True
            import_config.import_inertia_tensor = False
            # import_config.distance_scale = 100
            import_config.fix_base = False
            import_config.make_default_prim = True
            import_config.create_physics_scene = True
            omni.kit.commands.execute(
                "URDFParseAndImportFile",
                urdf_path=self._extension_path + "/data/urdf/robots/mooncake/urdf/mooncake.urdf",
                import_config=import_config,
            )

            viewport = omni.kit.viewport.get_default_viewport_window()
            viewport.set_camera_position("/OmniverseKit_Persp", -51, 63, 25, True)
            viewport.set_camera_target("/OmniverseKit_Persp", 220, -218, -160, True)
            stage = omni.usd.get_context().get_stage()
            scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))
            scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
            scene.CreateGravityMagnitudeAttr().Set(981.0)

            result, plane_path = omni.kit.commands.execute(
                "AddGroundPlaneCommand",
                stage=stage,
                planePath="/groundPlane",
                axis="Z",
                size=1500.0,
                position=Gf.Vec3f(0, 0, -25),
                color=Gf.Vec3f(0.5),
            )
            # make sure the ground plane is under root prim and not robot
            omni.kit.commands.execute(
                "MovePrimCommand", path_from=plane_path, path_to="/groundPlane", keep_world_transform=True
            )

            distantLight = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
            distantLight.CreateIntensityAttr(500)

    def _on_config_robot(self):
        stage = omni.usd.get_context().get_stage()
        # Make all rollers spin freely by removing extra drive API
        for wheel_index in range(3):
            for plate_index in range(2):
                for roller_index in range(9):
                    prim_path = "/mooncake/wheel_{}/roller_{}_{}_{}_joint".format(wheel_index, wheel_index, plate_index, roller_index)
                    prim = stage.GetPrimAtPath(prim_path)
                    omni.kit.commands.execute(
                        "UnapplyAPISchemaCommand",
                        api=UsdPhysics.DriveAPI,
                        prim=prim,
                        api_prefix="drive",
                        multiple_api_token="angular",
                    )

    def _on_config_drives(self):
        # self._on_config_robot()  # make sure drives are configured first
        stage = omni.usd.get_context().get_stage()
        # set each axis to spin at a rate of 1 rad/s
        axle_0 = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath("/mooncake/base_plate/wheel_0_joint"), "angular")
        axle_1 = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath("/mooncake/base_plate/wheel_1_joint"), "angular")
        axle_2 = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath("/mooncake/base_plate/wheel_2_joint"), "angular")

        omega = velocity2omega(0, 0.1, 0)
        print(omega)
        set_drive_parameters(axle_0, "velocity", math.degrees(omega[0]), 0, math.radians(1e7))
        set_drive_parameters(axle_1, "velocity", math.degrees(omega[1]), 0, math.radians(1e7))
        set_drive_parameters(axle_2, "velocity", math.degrees(omega[2]), 0, math.radians(1e7))
        # set_drive_parameters(axle_0, "effort", math.degrees(omega[0]), 0, math.radians(1e7))
        # set_drive_parameters(axle_1, "effort", math.degrees(omega[1]), 0, math.radians(1e7))
        # set_drive_parameters(axle_2, "effort", math.degrees(omega[2]), 0, math.radians(1e7))
