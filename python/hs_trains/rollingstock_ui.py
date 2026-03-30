#!/usr/bin/env python3
"""NiceGUI editor for RailML 3.3 rollingstock data."""

import io
import xml.etree.ElementTree as ET
from decimal import Decimal
from xml.etree.ElementTree import ElementTree, fromstring, indent

import typer
from nicegui import events, ui

from hs_trains.model.rollingstock import (
    Brakes, BrakeSystem, DaviesFormula, DecelerationCurve, Designator,
    DrivingResistance, DrivingResistanceInfo, Engine, Formation, Formations,
    PowerMode, RailML, Rollingstock, TractiveEffortCurve, TractionData,
    TractionDetails, TractionInfo, TrainDrivingResistance, TrainEngine,
    TrainOrder, TrainTractionMode, Value, ValueLine, ValueTable, Vehicle,
    VehiclePart, Vehicles,
)
from hs_trains.utils import validate_xml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dec(v) -> Decimal | None:
    if v is None:
        return None
    try:
        return Decimal(str(v))
    except (ValueError, ArithmeticError):
        return None


def _pos(v) -> Decimal | None:
    """Return Decimal only when v > 0."""
    d = _dec(v)
    return d if d and d > 0 else None


def _make_xml(vehicles: list[Vehicle], formations: list[Formation]) -> str:
    ET.register_namespace("rail3", "https://www.railml.org/schemas/3.3")
    railml = RailML(
        rollingstock=Rollingstock(
            vehicles=Vehicles(vehicles=vehicles) if vehicles else None,
            formations=Formations(formations=formations) if formations else None,
        )
    )
    xml_str = railml.to_xml(encoding="unicode", exclude_none=True)
    root = fromstring(xml_str)
    indent(root, space="  ")
    buf = io.StringIO()
    ElementTree(root).write(buf, encoding="unicode", xml_declaration=True)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# App state  (module-level — single-user local tool)
# ---------------------------------------------------------------------------

class _State:
    def __init__(self) -> None:
        self.vehicles: list[Vehicle] = []
        self.formations: list[Formation] = []
        self.vehicle_idx: int | None = None
        self.formation_idx: int | None = None

state = _State()


# ---------------------------------------------------------------------------
# Vehicle draft  (holds form values between refreshes)
# ---------------------------------------------------------------------------

class _VehicleDraft:
    def __init__(self) -> None:
        self.reset(None, 0)

    def reset(self, v: Vehicle | None, n: int) -> None:
        vid = v.id if v else f"vehicle_{n + 1:03d}"
        self.id         = vid
        self.speed      = float(v.speed or 120)                 if v else 120.0
        self.length     = float(v.length or 0)                  if v else 0.0
        self.tare       = float(v.tare_weight or 0)             if v else 0.0
        self.brutto     = float(v.brutto_weight or 0)           if v else 0.0
        self.driven     = float(v.number_of_driven_axles or 0)  if v else 0.0
        self.non_driven = float(v.number_of_non_driven_axles or 0) if v else 0.0
        self.adhesion   = float(v.adhesion_weight or 0)         if v else 0.0
        self.rot_mass   = float(v.rotating_mass_factor or 1.0)  if v else 1.0
        # engine
        pm = (v.engines[0].power_modes[0]
              if v and v.engines and v.engines[0].power_modes else None)
        self.has_engine = pm is not None
        self.mode       = pm.mode if pm else "diesel"
        ti = pm.traction_data.info if pm and pm.traction_data else None
        self.max_te     = float(ti.max_tractive_effort) if ti else 270_000.0
        self.power      = float(ti.tractive_power)      if ti else 2_420_000.0
        te_obj = (pm.traction_data.details.tractive_effort
                  if pm and pm.traction_data and pm.traction_data.details else None)
        self.te_pts = (
            [{"x": float(l.x_value), "y": float(l.values[0].y_value)}
             for l in te_obj.value_table.value_lines]
            if te_obj else [{"x": float(i * 30), "y": 0.0} for i in range(4)]
        )
        # brakes
        brk = v.brakes[0] if v and v.brakes else None
        dc  = brk.deceleration_table if brk else None
        self.decel_pts = (
            [{"x": float(l.x_value), "y": float(l.values[0].y_value)}
             for l in dc.value_table.value_lines]
            if dc else [{"x": float(i * 40), "y": 0.0} for i in range(4)]
        )
        bs = brk.vehicle_brakes[0] if brk and brk.vehicle_brakes else None
        self.brake_type = (bs.brake_type or "(none)") if bs else "(none)"
        self.reg_pct    = float(bs.regular_brake_percentage or 0)   if bs else 0.0
        self.emg_pct    = float(bs.emergency_brake_percentage or 0) if bs else 0.0
        # resistance
        dr = v.driving_resistance if v else None
        self.has_res    = dr is not None
        self.air_drag   = float(dr.info.air_drag_coefficient) if dr and dr.info else 0.80
        self.cross_sec  = float(dr.info.cross_section_area)   if dr and dr.info else 9.50
        self.rolling    = float(dr.info.rolling_resistance)   if dr and dr.info else 1.50
        self.tunnel_v   = float(dr.tunnel_factor or 1.5)      if dr else 1.5
        # dynamic lists
        self.parts = (
            [{"id": p.id, "order": float(p.part_order), "cat": p.category or "(none)"}
             for p in v.vehicle_parts]
            if v and v.vehicle_parts
            else [{"id": f"{vid}_part_01", "order": 1.0, "cat": "(none)"}]
        )
        self.designators = (
            [{"reg": d.register_name, "entry": d.entry} for d in v.designators]
            if v and v.designators else []
        )

    def build(self) -> Vehicle:
        te_curve = None
        if self.has_engine and any(p["y"] != 0.0 for p in self.te_pts):
            te_curve = TractiveEffortCurve(
                value_table=ValueTable(
                    x_value_name="speed", x_value_unit="km/h",
                    y_value_name="tractiveEffort", y_value_unit="N",
                    value_lines=[
                        ValueLine(x_value=Decimal(str(p["x"])),
                                  values=[Value(y_value=Decimal(str(p["y"])))])
                        for p in self.te_pts
                    ],
                )
            )
        engine = None
        if self.has_engine:
            engine = Engine(power_modes=[PowerMode(
                mode=self.mode, is_primary_mode=True,
                traction_data=TractionData(
                    info=TractionInfo(
                        max_tractive_effort=Decimal(str(self.max_te)),
                        tractive_power=Decimal(str(self.power)),
                    ) if self.max_te > 0 else None,
                    details=TractionDetails(tractive_effort=te_curve) if te_curve else None,
                ),
            )])
        dc_curve = None
        if any(p["y"] != 0.0 for p in self.decel_pts):
            dc_curve = DecelerationCurve(
                value_table=ValueTable(
                    x_value_name="speed", x_value_unit="km/h",
                    y_value_name="deceleration", y_value_unit="m/s/s",
                    value_lines=[
                        ValueLine(x_value=Decimal(str(p["x"])),
                                  values=[Value(y_value=Decimal(str(p["y"])))])
                        for p in self.decel_pts
                    ],
                )
            )
        bsys = []
        if self.brake_type != "(none)" or self.reg_pct > 0 or self.emg_pct > 0:
            bsys.append(BrakeSystem(
                brake_type=self.brake_type if self.brake_type != "(none)" else None,
                regular_brake_percentage=_pos(self.reg_pct),
                emergency_brake_percentage=_pos(self.emg_pct),
            ))
        brakes = ([Brakes(vehicle_brakes=bsys, deceleration_table=dc_curve)]
                  if (bsys or dc_curve) else [])
        dr = None
        if self.has_res:
            dr = DrivingResistance(
                tunnel_factor=_dec(self.tunnel_v),
                info=DrivingResistanceInfo(
                    air_drag_coefficient=Decimal(str(self.air_drag)),
                    cross_section_area=Decimal(str(self.cross_sec)),
                    rolling_resistance=Decimal(str(self.rolling)),
                ),
            )
        return Vehicle(
            id=self.id,
            speed=_pos(self.speed), length=_pos(self.length),
            tare_weight=_pos(self.tare), brutto_weight=_pos(self.brutto),
            number_of_driven_axles=int(self.driven) or None,
            number_of_non_driven_axles=int(self.non_driven) or None,
            adhesion_weight=_pos(self.adhesion),
            rotating_mass_factor=_pos(self.rot_mass),
            designators=[
                Designator(register_name=d["reg"], entry=d["entry"])
                for d in self.designators if d["reg"] and d["entry"]
            ],
            vehicle_parts=[
                VehiclePart(id=p["id"], part_order=int(p["order"]),
                            category=None if p["cat"] == "(none)" else p["cat"])
                for p in self.parts
            ],
            engines=[engine] if engine else [],
            brakes=brakes,
            driving_resistance=dr,
        )


vd = _VehicleDraft()


# ---------------------------------------------------------------------------
# Formation draft
# ---------------------------------------------------------------------------

class _FormationDraft:
    def __init__(self) -> None:
        self.reset(None, 0)

    def reset(self, f: Formation | None, n: int) -> None:
        self.id       = f.id if f else f"formation_{n + 1:03d}"
        self.speed    = float(f.speed or 120)        if f else 120.0
        self.length   = float(f.length or 0)         if f else 0.0
        self.tare     = float(f.tare_weight or 0)    if f else 0.0
        self.brutto   = float(f.brutto_weight or 0)  if f else 0.0
        self.n_axles  = float(f.number_of_axles or 0)  if f else 0.0
        self.n_wagons = float(f.number_of_wagons or 0) if f else 0.0
        te = f.train_engines[0] if f and f.train_engines else None
        self.has_engine = te is not None
        self.max_acc  = float(te.max_acceleration or 0.4)   if te else 0.4
        self.mean_acc = float(te.mean_acceleration or 0.25) if te else 0.25
        tm = te.traction_mode if te else None
        self.f_mode   = tm.mode if tm else "diesel"
        tr = f.train_resistance if f else None
        self.has_davies = tr is not None
        df = tr.davies_formula_factors if tr else None
        self.a_val    = float(df.constant_factor_a)               if df else 3800.0
        self.b_val    = float(df.speed_dependent_factor_b)        if df else 45.0
        self.c_val    = float(df.square_speed_dependent_factor_c) if df else 2.5
        self.tunnel_f = float(tr.tunnel_factor or 1.8)            if tr else 1.8
        self.orders = (
            [{"pos": float(o.order_number), "vref": o.vehicle_ref, "ori": o.orientation}
             for o in f.train_orders]
            if f and f.train_orders else []
        )
        self.designators = (
            [{"reg": d.register_name, "entry": d.entry} for d in f.designators]
            if f and f.designators else []
        )

    def build(self) -> Formation:
        te = None
        if self.has_engine:
            te = TrainEngine(
                max_acceleration=_dec(self.max_acc),
                mean_acceleration=_dec(self.mean_acc),
                traction_mode=TrainTractionMode(mode=self.f_mode, is_primary_mode=True),
            )
        tr = None
        if self.has_davies:
            tr = TrainDrivingResistance(
                davies_formula_factors=DaviesFormula(
                    constant_factor_a=Decimal(str(self.a_val)),
                    speed_dependent_factor_b=Decimal(str(self.b_val)),
                    square_speed_dependent_factor_c=Decimal(str(self.c_val)),
                    mass_dependent=False,
                ),
                tunnel_factor=_dec(self.tunnel_f),
            )
        return Formation(
            id=self.id,
            speed=_pos(self.speed), length=_pos(self.length),
            tare_weight=_pos(self.tare), brutto_weight=_pos(self.brutto),
            number_of_axles=int(self.n_axles) or None,
            number_of_wagons=int(self.n_wagons) or None,
            designators=[
                Designator(register_name=d["reg"], entry=d["entry"])
                for d in self.designators if d["reg"] and d["entry"]
            ],
            train_orders=[
                TrainOrder(order_number=int(o["pos"]), vehicle_ref=o["vref"],
                           orientation=o["ori"])
                for o in self.orders if o["vref"]
            ],
            train_engines=[te] if te else [],
            train_resistance=tr,
        )


fd = _FormationDraft()


# ---------------------------------------------------------------------------
# Reusable UI helpers  (container-clear pattern — no session state needed)
# ---------------------------------------------------------------------------

_PART_CATS = ["(none)", "locomotive", "motorCoach", "passengerCoach",
              "freightWagon", "cabCoach", "booster"]


def _render_curve(pts: list[dict], x_label: str, y_label: str,
                  container: ui.column) -> None:
    """Clear and re-render a speed→value curve inside *container*."""
    container.clear()
    with container:
        for i, pt in enumerate(pts):
            def _rm(_, i=i):
                pts.pop(i)
                _render_curve(pts, x_label, y_label, container)
            with ui.row().classes("items-center gap-2"):
                ui.number(x_label, value=pt["x"]).bind_value(pt, "x").classes("w-36")
                ui.number(y_label, value=pt["y"]).bind_value(pt, "y").classes("w-36")
                ui.button("✕", on_click=_rm).props("flat dense color=negative")
        ui.button("+ Row", on_click=lambda: (
            pts.append({"x": 0.0, "y": 0.0}),
            _render_curve(pts, x_label, y_label, container),
        )).props("flat dense")


def _render_designators(items: list[dict], container: ui.column) -> None:
    container.clear()
    with container:
        for i, d in enumerate(items):
            def _rm(_, i=i):
                items.pop(i)
                _render_designators(items, container)
            with ui.row().classes("items-center gap-2"):
                ui.input("Register", value=d["reg"]).bind_value(d, "reg").classes("w-32")
                ui.input("Entry",    value=d["entry"]).bind_value(d, "entry").classes("w-48")
                ui.button("✕", on_click=_rm).props("flat dense color=negative")
        ui.button("+ Designator", on_click=lambda: (
            items.append({"reg": "", "entry": ""}),
            _render_designators(items, container),
        )).props("flat dense")


def _render_parts(parts: list[dict], vehicle_id: str, container: ui.column) -> None:
    container.clear()
    with container:
        for i, p in enumerate(parts):
            def _rm(_, i=i):
                parts.pop(i)
                _render_parts(parts, vehicle_id, container)
            with ui.row().classes("items-center gap-2"):
                ui.input("Part ID", value=p["id"]).bind_value(p, "id").classes("w-48")
                ui.number("Order", value=p["order"], step=1).bind_value(p, "order").classes("w-20")
                ui.select(_PART_CATS, value=p["cat"]).bind_value(p, "cat").classes("w-40")
                ui.button("✕", on_click=_rm).props("flat dense color=negative")
        ui.button("+ Part", on_click=lambda: (
            parts.append({
                "id": f"{vehicle_id}_part_{len(parts) + 1:02d}",
                "order": float(len(parts) + 1),
                "cat": "(none)",
            }),
            _render_parts(parts, vehicle_id, container),
        )).props("flat dense")


# ---------------------------------------------------------------------------
# Sidebar lists
# ---------------------------------------------------------------------------

@ui.refreshable
def vehicle_list() -> None:
    for i, v in enumerate(state.vehicles):
        is_sel = state.vehicle_idx == i
        def _sel(_, i=i):
            state.vehicle_idx = i
            vd.reset(state.vehicles[i], len(state.vehicles))
            vehicle_form.refresh()
            vehicle_list.refresh()

        def _del(_, i=i):
            state.vehicles.pop(i)
            if state.vehicle_idx == i:
                state.vehicle_idx = None
                vd.reset(None, len(state.vehicles))
                vehicle_form.refresh()
            elif state.vehicle_idx is not None and state.vehicle_idx > i:
                state.vehicle_idx -= 1
            vehicle_list.refresh()

        with ui.row().classes("items-center w-full gap-0"):
            ui.button(v.id, on_click=_sel).props(
                f"flat dense align=left {'color=primary' if is_sel else ''}"
            ).classes("flex-1 text-left")
            ui.button("✕", on_click=_del).props("flat dense color=negative")


@ui.refreshable
def formation_list() -> None:
    for i, f in enumerate(state.formations):
        is_sel = state.formation_idx == i
        def _sel(_, i=i):
            state.formation_idx = i
            fd.reset(state.formations[i], len(state.formations))
            formation_form.refresh()
            formation_list.refresh()

        def _del(_, i=i):
            state.formations.pop(i)
            if state.formation_idx == i:
                state.formation_idx = None
                fd.reset(None, len(state.formations))
                formation_form.refresh()
            elif state.formation_idx is not None and state.formation_idx > i:
                state.formation_idx -= 1
            formation_list.refresh()

        with ui.row().classes("items-center w-full gap-0"):
            ui.button(f.id, on_click=_sel).props(
                f"flat dense align=left {'color=primary' if is_sel else ''}"
            ).classes("flex-1 text-left")
            ui.button("✕", on_click=_del).props("flat dense color=negative")


# ---------------------------------------------------------------------------
# Vehicle form
# ---------------------------------------------------------------------------

@ui.refreshable
def vehicle_form() -> None:
    v = state.vehicles[state.vehicle_idx] if state.vehicle_idx is not None else None
    label = f"Editing: {v.id}" if v else "New Vehicle"

    with ui.card().classes("w-full"):
        ui.label(label).classes("text-h6")

        with ui.tabs().classes("w-full") as tabs:
            t_basic  = ui.tab("Basic")
            t_parts  = ui.tab("Parts & Designators")
            t_engine = ui.tab("Engine")
            t_brakes = ui.tab("Brakes")
            t_res    = ui.tab("Resistance")

        with ui.tab_panels(tabs, value=t_basic).classes("w-full"):

            with ui.tab_panel(t_basic):
                ui.input("Vehicle ID", value=vd.id).bind_value(vd, "id").classes("w-full")
                with ui.grid(columns=2).classes("w-full gap-2"):
                    ui.number("Max speed (km/h)",    value=vd.speed,      step=1).bind_value(vd, "speed")
                    ui.number("Length (m)",          value=vd.length               ).bind_value(vd, "length")
                    ui.number("Tare weight (t)",     value=vd.tare                 ).bind_value(vd, "tare")
                    ui.number("Brutto weight (t)",   value=vd.brutto               ).bind_value(vd, "brutto")
                    ui.number("Driven axles",        value=vd.driven,     step=1   ).bind_value(vd, "driven")
                    ui.number("Non-driven axles",    value=vd.non_driven, step=1   ).bind_value(vd, "non_driven")
                    ui.number("Adhesion weight (t)", value=vd.adhesion             ).bind_value(vd, "adhesion")
                    ui.number("Rotating mass factor",value=vd.rot_mass             ).bind_value(vd, "rot_mass")

            with ui.tab_panel(t_parts):
                ui.label("Designators").classes("text-subtitle2")
                des_cont = ui.column().classes("w-full")
                _render_designators(vd.designators, des_cont)

                ui.separator().classes("my-2")
                ui.label("Vehicle Parts").classes("text-subtitle2")
                parts_cont = ui.column().classes("w-full")

                _render_parts(vd.parts, vd.id, parts_cont)

            with ui.tab_panel(t_engine):
                ui.switch("Vehicle has an engine", value=vd.has_engine).bind_value(vd, "has_engine")
                with ui.column().classes("w-full").bind_visibility_from(vd, "has_engine"):
                    ui.select(["diesel", "electric", "battery"], label="Power mode",
                              value=vd.mode).bind_value(vd, "mode")
                    with ui.grid(columns=2).classes("w-full gap-2"):
                        ui.number("Max tractive effort (N)", value=vd.max_te).bind_value(vd, "max_te")
                        ui.number("Tractive power (W)",      value=vd.power ).bind_value(vd, "power")
                    ui.label("Tractive effort curve  speed (km/h) → effort (N)").classes("text-caption mt-2")
                    te_cont = ui.column().classes("w-full")
                    _render_curve(vd.te_pts, "Speed (km/h)", "Force (N)", te_cont)

            with ui.tab_panel(t_brakes):
                ui.label("Deceleration curve  speed (km/h) → decel (m/s²)").classes("text-caption")
                decel_cont = ui.column().classes("w-full")
                _render_curve(vd.decel_pts, "Speed (km/h)", "Decel (m/s²)", decel_cont)
                ui.separator().classes("my-2")
                ui.label("Brake system").classes("text-subtitle2")
                with ui.row().classes("gap-4"):
                    ui.select(["(none)", "P", "G", "R"], label="Brake type",
                              value=vd.brake_type).bind_value(vd, "brake_type")
                    ui.number("Regular brake %",   value=vd.reg_pct).bind_value(vd, "reg_pct")
                    ui.number("Emergency brake %", value=vd.emg_pct).bind_value(vd, "emg_pct")

            with ui.tab_panel(t_res):
                ui.switch("Specify driving resistance", value=vd.has_res).bind_value(vd, "has_res")
                with ui.column().classes("w-full").bind_visibility_from(vd, "has_res"):
                    with ui.grid(columns=2).classes("w-full gap-2"):
                        ui.number("Air drag coeff (Cd)",       value=vd.air_drag  ).bind_value(vd, "air_drag")
                        ui.number("Cross section area (m²)",   value=vd.cross_sec ).bind_value(vd, "cross_sec")
                        ui.number("Rolling resistance (N/kN)", value=vd.rolling   ).bind_value(vd, "rolling")
                        ui.number("Tunnel factor",             value=vd.tunnel_v  ).bind_value(vd, "tunnel_v")

        def _save():
            try:
                v = vd.build()
            except Exception as ex:
                ui.notify(f"Build error: {ex}", color="negative", timeout=5000)
                return
            if state.vehicle_idx is not None:
                state.vehicles[state.vehicle_idx] = v
            else:
                state.vehicles.append(v)
                state.vehicle_idx = len(state.vehicles) - 1
            vehicle_list.refresh()
            ui.notify(f"Saved '{v.id}'", color="positive")

        ui.button("Save Vehicle", on_click=_save).props("color=primary").classes("mt-4")


# ---------------------------------------------------------------------------
# Formation form
# ---------------------------------------------------------------------------

@ui.refreshable
def formation_form() -> None:
    f = state.formations[state.formation_idx] if state.formation_idx is not None else None
    label = f"Editing: {f.id}" if f else "New Formation"

    with ui.card().classes("w-full"):
        ui.label(label).classes("text-h6")

        with ui.tabs().classes("w-full") as tabs:
            t_basic  = ui.tab("Basic")
            t_order  = ui.tab("Train Order")
            t_engine = ui.tab("Engine")
            t_res    = ui.tab("Resistance")

        with ui.tab_panels(tabs, value=t_basic).classes("w-full"):

            with ui.tab_panel(t_basic):
                ui.input("Formation ID", value=fd.id).bind_value(fd, "id").classes("w-full")
                with ui.grid(columns=2).classes("w-full gap-2"):
                    ui.number("Max speed (km/h)",   value=fd.speed,    step=1).bind_value(fd, "speed")
                    ui.number("Total length (m)",   value=fd.length          ).bind_value(fd, "length")
                    ui.number("Tare weight (t)",    value=fd.tare            ).bind_value(fd, "tare")
                    ui.number("Brutto weight (t)",  value=fd.brutto          ).bind_value(fd, "brutto")
                    ui.number("Number of axles",    value=fd.n_axles,  step=1).bind_value(fd, "n_axles")
                    ui.number("Number of wagons",   value=fd.n_wagons, step=1).bind_value(fd, "n_wagons")
                ui.separator().classes("my-2")
                ui.label("Designators").classes("text-subtitle2")
                des_cont = ui.column().classes("w-full")
                _render_designators(fd.designators, des_cont)

            with ui.tab_panel(t_order):
                if not state.vehicles:
                    ui.label("No vehicles defined yet — add vehicles first.").classes("text-orange")
                else:
                    vehicle_ids = [v.id for v in state.vehicles]
                    orders_cont = ui.column().classes("w-full")

                    def _render_orders():
                        orders_cont.clear()
                        with orders_cont:
                            with ui.row().classes("gap-2 text-caption text-grey"):
                                ui.label("Pos").classes("w-16")
                                ui.label("Vehicle").classes("w-52")
                                ui.label("Orientation").classes("w-32")
                            for i, o in enumerate(fd.orders):
                                # ensure vref is valid
                                if o["vref"] not in vehicle_ids:
                                    o["vref"] = vehicle_ids[0]
                                def _rm(_, i=i):
                                    fd.orders.pop(i)
                                    _render_orders()
                                with ui.row().classes("items-center gap-2"):
                                    ui.number("", value=o["pos"], step=1).bind_value(o, "pos").classes("w-16")
                                    ui.select(vehicle_ids, value=o["vref"]).bind_value(o, "vref").classes("w-52")
                                    ui.select(["normal", "reverse"], value=o["ori"]).bind_value(o, "ori").classes("w-32")
                                    ui.button("✕", on_click=_rm).props("flat dense color=negative")
                            ui.button("+ Add to order", on_click=lambda: (
                                fd.orders.append({
                                    "pos": float(len(fd.orders) + 1),
                                    "vref": vehicle_ids[0],
                                    "ori": "normal",
                                }),
                                _render_orders(),
                            )).props("flat dense")

                    _render_orders()

            with ui.tab_panel(t_engine):
                ui.switch("Specify train engine", value=fd.has_engine).bind_value(fd, "has_engine")
                with ui.column().classes("w-full").bind_visibility_from(fd, "has_engine"):
                    with ui.grid(columns=2).classes("w-full gap-2"):
                        ui.number("Max acceleration (m/s²)",  value=fd.max_acc ).bind_value(fd, "max_acc")
                        ui.number("Mean acceleration (m/s²)", value=fd.mean_acc).bind_value(fd, "mean_acc")
                    ui.select(["diesel", "electric", "battery"], label="Traction mode",
                              value=fd.f_mode).bind_value(fd, "f_mode")

            with ui.tab_panel(t_res):
                ui.switch("Specify Davies formula", value=fd.has_davies).bind_value(fd, "has_davies")
                with ui.column().classes("w-full").bind_visibility_from(fd, "has_davies"):
                    ui.label("R(N) = A + B·v + C·v²  (v in km/h)").classes("text-caption")
                    with ui.grid(columns=2).classes("w-full gap-2"):
                        ui.number("A (N)",          value=fd.a_val  ).bind_value(fd, "a_val")
                        ui.number("B (N·h/km)",     value=fd.b_val  ).bind_value(fd, "b_val")
                        ui.number("C (N·h²/km²)",   value=fd.c_val  ).bind_value(fd, "c_val")
                        ui.number("Tunnel factor",  value=fd.tunnel_f).bind_value(fd, "tunnel_f")

        def _save():
            try:
                f = fd.build()
            except Exception as ex:
                ui.notify(f"Build error: {ex}", color="negative", timeout=5000)
                return
            if state.formation_idx is not None:
                state.formations[state.formation_idx] = f
            else:
                state.formations.append(f)
                state.formation_idx = len(state.formations) - 1
            formation_list.refresh()
            ui.notify(f"Saved '{f.id}'", color="positive")

        ui.button("Save Formation", on_click=_save).props("color=primary").classes("mt-4")


MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


# ---------------------------------------------------------------------------
# Import / Export dialog
# ---------------------------------------------------------------------------

def _open_import_export() -> None:
    with ui.dialog() as dlg, ui.card().classes("w-full max-w-2xl"):
        ui.label("Import / Export").classes("text-h6")

        ui.label("Import XML").classes("text-subtitle2 mt-2")

        async def _on_upload(e: events.UploadEventArguments) -> None:
            try:
                data = await e.file.read()
                if len(data) > MAX_UPLOAD_BYTES:
                    ui.notify(
                        f"File too large ({len(data) // 1024} KB); limit is {MAX_UPLOAD_BYTES // 1024 // 1024} MB.",
                        color="negative",
                    )
                    return
                railml = RailML.from_xml(data)
                state.vehicles = (
                    railml.rollingstock.vehicles.vehicles
                    if railml.rollingstock and railml.rollingstock.vehicles else []
                )
                state.formations = (
                    railml.rollingstock.formations.formations
                    if railml.rollingstock and railml.rollingstock.formations else []
                )
                state.vehicle_idx = None
                state.formation_idx = None
                vd.reset(None, 0)
                fd.reset(None, 0)
                vehicle_list.refresh()
                formation_list.refresh()
                vehicle_form.refresh()
                formation_form.refresh()
                ui.notify(
                    f"Loaded {len(state.vehicles)} vehicle(s), "
                    f"{len(state.formations)} formation(s).",
                    color="positive",
                )
                dlg.close()
            except Exception as ex:
                ui.notify(f"Parse error: {ex}", color="negative", timeout=8000)

        ui.upload(on_upload=_on_upload, label="Select RailML XML").props('accept=".xml"').classes("w-full")

        ui.separator().classes("my-2")
        ui.label("Export XML").classes("text-subtitle2")

        def _export() -> None:
            if not state.vehicles and not state.formations:
                ui.notify("Nothing to export.", color="warning")
                return
            try:
                xml_str = _make_xml(state.vehicles, state.formations)
            except Exception as ex:
                ui.notify(f"XML generation failed: {ex}", color="negative")
                return
            try:
                errors = validate_xml(xml_str)
            except FileNotFoundError as exc:
                ui.notify(str(exc), color="warning", timeout=6000)
                errors = []
            if errors:
                ui.notify(
                    f"Validation: {len(errors)} error(s) — {errors[0][:120]}",
                    color="negative", multi_line=True, timeout=8000,
                )
            else:
                ui.notify("Schema valid (RailML 3.3)", color="positive")
            ui.download(xml_str.encode(), "rollingstock.xml")

        ui.button("Validate & Download", on_click=_export).props("color=primary")
        ui.button("Close", on_click=dlg.close).props("flat")

    dlg.open()


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

def build_page() -> None:
    with ui.header().classes("bg-blue-800 text-white items-center px-4"):
        ui.label("RailML 3.3 Rollingstock Builder").classes("text-h6")
        ui.space()
        ui.button("Import / Export", on_click=_open_import_export).props("flat color=white")

    with ui.row().classes("w-full h-full no-wrap"):
        # ── left sidebar ──────────────────────────────────────────────────
        with ui.card().classes("w-56 shrink-0 rounded-none h-full q-pa-sm"):

            with ui.row().classes("items-center justify-between w-full"):
                ui.label("VEHICLES").classes("text-caption text-grey-6")
                def _new_vehicle():
                    state.vehicle_idx = None
                    vd.reset(None, len(state.vehicles))
                    vehicle_form.refresh()
                ui.button("+ New", on_click=_new_vehicle).props("flat dense")
            vehicle_list()

            ui.separator().classes("my-2")

            with ui.row().classes("items-center justify-between w-full"):
                ui.label("FORMATIONS").classes("text-caption text-grey-6")
                def _new_formation():
                    state.formation_idx = None
                    fd.reset(None, len(state.formations))
                    formation_form.refresh()
                ui.button("+ New", on_click=_new_formation).props("flat dense")
            formation_list()

        # ── right panel ───────────────────────────────────────────────────
        with ui.column().classes("flex-1 p-4 overflow-auto"):
            with ui.tabs().classes("w-full") as main_tabs:
                t_veh  = ui.tab("Vehicle")
                t_form = ui.tab("Formation")
            with ui.tab_panels(main_tabs, value=t_veh).classes("w-full"):
                with ui.tab_panel(t_veh):
                    vehicle_form()
                with ui.tab_panel(t_form):
                    formation_form()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _run(port: int = typer.Option(8080, help="Port to serve the UI on")) -> None:
    """Start the RailML 3.3 Rollingstock Builder UI."""
    @ui.page("/")
    def index() -> None:
        vd.reset(None, 0)
        fd.reset(None, 0)
        build_page()

    ui.run(title="RailML 3.3 Rollingstock Builder", port=port, reload=False)


def launch() -> None:
    """Entry point for the `rollingstock-editor` console script."""
    typer.run(_run)


if __name__ == "__main__":
    launch()
