import streamlit as st
import numpy as np
import plotly.graph_objects as go

# PAGE CONFIG =============================================================================

st.set_page_config(page_title="G.L.I.D.E.R. Simulator", layout="wide")

# HARDCODED VALUES ========================================================================

G_FPS2 = 32.174            # Gravity acceleration [ft/s^2]
RHO_AIR = 0.002377         # Air density at standard atmosphere [slugs/ft^3]
MU_K = 0.05                # Kinetic rail friction coefficient
C_DC = 0.80                # Carriage drag coefficient
A_CARRIAGE = 0.0861        # Carriage frontal area [ft^2]
DT = 0.0008                # Integration time step [s]
F_SPRING_LBF = 33.7        # Constant spring force [lbf]
W_CARRIAGE_LB = 0.551      # Carriage weight [lb]

# SIDEBAR =================================================================================

st.sidebar.subheader("1. Glider Parameters")
w_glider_oz = st.sidebar.number_input("Glider Weight (oz)", min_value=0.5, max_value=20.0, value=1.76, step=0.1)
w_glider_lb = w_glider_oz / 16.0
m_glider_slug = w_glider_lb / G_FPS2

st.sidebar.subheader("2. Aerodynamic Variables")
wingspan_in = st.sidebar.number_input("Wingspan (in)", min_value=5.0, max_value=48.0, value=24.0, step=1.0)
wingspan_ft = wingspan_in / 12.0

s_wing_sqft = st.sidebar.number_input("Wing Planform Area (ft²)", min_value=0.1, max_value=3.0, value=0.484, step=0.05)
c_l = st.sidebar.number_input("Lift Coefficient (C_L)", min_value=0.10, max_value=1.50, value=0.50, step=0.05)

aspect_ratio = (wingspan_ft ** 2) / s_wing_sqft
c_d0 = 0.03
c_di = (c_l ** 2) / (np.pi * aspect_ratio * 0.8)
c_d = c_d0 + c_di

# btn_generate_contour = st.sidebar.button("Refresh Contour Map if ")

st.sidebar.subheader("3. Launch Setup Sliders")
d_draw_in = st.sidebar.slider("Drawback Distance (in)", min_value=0.0, max_value=36.0, value=20.0, step=0.5)
launch_angle_deg = st.sidebar.slider("Launch Pitch Angle (°)", min_value=0.0, max_value=20.0, value=5.0, step=0.5)

st.sidebar.subheader("4. Launcher Platform Specs")
st.sidebar.text(f"Spring Force: {F_SPRING_LBF:.1f} lbf")
st.sidebar.text(f"Carriage Weight: {W_CARRIAGE_LB:.2f} lb")

# CALCULATOR ==============================================================================

def simulate_rail_launch(w_glider_lb, d_draw_in, theta_deg):
    """Calculates carriage acceleration along the guide rail and final exit velocity."""
    w_total_lb = w_glider_lb + W_CARRIAGE_LB
    m_total_slug = w_total_lb / G_FPS2
    theta_rad = np.radians(theta_deg)
    d_draw_ft = d_draw_in / 12.0

    if d_draw_ft <= 0:
        return 0.0, 0.0, 0.0, m_total_slug

    f_friction = MU_K * w_total_lb * np.cos(theta_rad)
    f_gravity = w_total_lb * np.sin(theta_rad)
    f_static_net = F_SPRING_LBF - f_friction - f_gravity

    if f_static_net <= 0:
        return 0.0, 0.0, 0.0, m_total_slug

    a_initial_fps2 = f_static_net / m_total_slug

    x_ft = 0.0
    v_fps = 0.0
    t_s = 0.0

    while x_ft < d_draw_ft and t_s < 2.0:
        f_drag = 0.5 * RHO_AIR * (v_fps**2) * C_DC * A_CARRIAGE
        f_net = F_SPRING_LBF - f_friction - f_gravity - f_drag

        if f_net <= 0 and v_fps <= 0.01:
            v_fps = 0.0
            break

        a_fps2 = f_net / m_total_slug
        v_fps += a_fps2 * DT
        x_ft += v_fps * DT
        t_s += DT

    return max(0.0, v_fps), a_initial_fps2, t_s, m_total_slug

def simulate_free_flight(v_launch_fps, theta_deg, m_glider_slug, s_wing_sqft, c_l_val, c_d_val):
    """Calculates 2D trajectory path (x, y) considering lift, drag, and gravity."""
    if v_launch_fps <= 0.1:
        return np.array([0.0]), np.array([0.0]), 0.0, 0.0, 0.0

    theta_rad = np.radians(theta_deg)

    x, y = 0.0, 1.0  # Initial launcher height off ground [ft]
    vx = v_launch_fps * np.cos(theta_rad)
    vy = v_launch_fps * np.sin(theta_rad)
    t = 0.0

    x_pts, y_pts = [x], [y]
    apex_ft = y
    t_apex = 0.0

    while y >= 0.0 and t < 15.0 and x < 300.0:
        v = np.hypot(vx, vy)
        if v < 0.01:
            break

        gamma = np.arctan2(vy, vx)
        q = 0.5 * RHO_AIR * (v**2)
        L = q * s_wing_sqft * c_l_val
        D = q * s_wing_sqft * c_d_val

        ax = (-D * np.cos(gamma) - L * np.sin(gamma)) / m_glider_slug
        ay = -G_FPS2 + (L * np.cos(gamma) - D * np.sin(gamma)) / m_glider_slug

        vx += ax * DT
        vy += ay * DT
        x += vx * DT
        y += vy * DT
        t += DT

        if y > apex_ft:
            apex_ft = y
            t_apex = t

        x_pts.append(x)
        y_pts.append(max(0.0, y))

    return np.array(x_pts), np.array(y_pts), apex_ft, t_apex, t

# EXECUTE =================================================================================

@st.cache_data
def generate_contour_grid(w_glider_lb, m_glider_slug, s_wing_sqft, c_l, c_d):
    """Generates contour map data once and caches it until weight/aero changes."""
    d_vals = np.linspace(0.0, 36.0, 25)
    a_vals = np.linspace(0.0, 20.0, 25)
    D_grid, A_grid = np.meshgrid(d_vals, a_vals)
    Z_range = np.zeros_like(D_grid)

    for i in range(D_grid.shape[0]):
        for j in range(D_grid.shape[1]):
            v_tmp, _, _, _ = simulate_rail_launch(w_glider_lb, D_grid[i, j], A_grid[i, j])
            x_tmp, _, _, _, _ = simulate_free_flight(v_tmp, A_grid[i, j], m_glider_slug, s_wing_sqft, c_l, c_d)
            Z_range[i, j] = x_tmp[-1]

    return d_vals, a_vals, Z_range

# Execute fast single trajectory simulation for current user inputs
v_launch, a_init, t_rail, m_total = simulate_rail_launch(w_glider_lb, d_draw_in, launch_angle_deg)
x_sim, y_sim, apex_ft, t_apex, t_flight = simulate_free_flight(
    v_launch, launch_angle_deg, m_glider_slug, s_wing_sqft, c_l, c_d
)

# OUTPUTS =================================================================================

col1, col2, col3, col4 = st.columns(4)
col1.metric("Exit Launch Velocity", f"{v_launch:.2f} ft/s")
col2.metric("Downrange Flight Distance", f"{x_sim[-1]:.2f} ft")
col3.metric("Apex Altitude", f"{apex_ft:.2f} ft")
col4.metric("Initial Acceleration", f"{a_init / G_FPS2:.1f} g")

st.divider()

col_left, col_right = st.columns([7, 5])

with col_left:
    fig_traj = go.Figure()

    fig_traj.add_shape(type="line", x0=0, y0=0, x1=max(max(x_sim) * 1.05, 10.0), y1=0,
                      line=dict(color="gray", width=2, dash="dash"))

    fig_traj.add_trace(go.Scatter(
        x=x_sim, y=y_sim,
        mode='lines',
        name=f'Flight Path',
        line=dict(color='#00D4FF', width=3)
    ))

    if apex_ft > 1.0 and len(x_sim) > 0:
        apex_idx = np.argmax(y_sim)
        fig_traj.add_trace(go.Scatter(
            x=[x_sim[apex_idx]], y=[y_sim[apex_idx]],
            mode='markers',
            name=f'Apex',
            marker=dict(color='#FF5555', size=10, symbol='diamond')
        ))

    fig_traj.update_layout(
        xaxis_title="Horizontal Distance (ft)",
        yaxis_title="Altitude (ft)",
        yaxis=dict(range=[0, max(max(y_sim) + 1.0, 5.0)], gridcolor='#333333'),
        xaxis=dict(range=[0, max(max(x_sim) + 2.0, 10.0)], gridcolor='#333333'),
        template="plotly_dark",
        paper_bgcolor="#111111",
        plot_bgcolor="#1E1E1E",
        height=400,
        margin=dict(l=20, r=20, t=20, b=20)
    )

    st.plotly_chart(fig_traj, use_container_width=True)

with col_right:
    d_vals, a_vals, Z_range = generate_contour_grid(w_glider_lb, m_glider_slug, s_wing_sqft, c_l, c_d)

    fig_contour = go.Figure(data=go.Contour(
        x=d_vals, y=a_vals, z=Z_range,
        colorscale='Viridis',
        colorbar=dict(title="Flight Range (ft)"),
        contours=dict(showlabels=True, labelfont=dict(size=10, color='white'))
    ))

    fig_contour.update_layout(
        xaxis_title="Drawback Distance (in)",
        yaxis_title="Pitch Angle (°)",
        template="plotly_dark",
        paper_bgcolor="#111111",
        plot_bgcolor="#1E1E1E",
        height=360,
        margin=dict(l=20, r=20, t=20, b=20)
    )

    st.plotly_chart(fig_contour, use_container_width=True)
