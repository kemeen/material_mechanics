import pytest
import material_mechanics as mm


@pytest.fixture(scope='module')
def ortho_mat():
    name = 'Orthotropic Material'

    stiffnesses = dict(e1=100000, e2=8000, e3=7000, g12=5000, g13=5000, g23=4000)
    poissons = dict(nu12=0.33, nu21=0.02, nu13=0.33, nu31=0.02, nu23=0.33, nu32=0.02)
    strengths = None
    density = 1.0

    return mm.orthotropic_material(name=name, stiffness=stiffnesses, poisson=poissons, strength=strengths, density=density)


@pytest.fixture(scope='module')
def gfk_mat():
    name = 'GFK'
    stiffnesses = dict(e1=45160, e2=14700, g12=5300)
    poissons = dict(nu12=0.3, nu23=0.38)
    strengths = ((1200, 900, None, None), (50, 170, 70, 70), (50, 170, 70, 70))
    density = 2.0

    return mm.transverse_isotropic_material(name=name, stiffness=stiffnesses, poisson=poissons, strength=strengths, density=density)


@pytest.fixture(scope='module')
def cfk_mat():
    name = 'CFK'
    stiffnesses = dict(e1=140000, e2=9000, g12=4600)
    poissons = dict(nu12=0.3, nu23=0.37)
    strengths = None
    density = 1.5

    material = mm.transverse_isotropic_material(name=name, stiffness=stiffnesses, poisson=poissons, strength=strengths, density=density)

    return material


@pytest.fixture(scope='module')
def fiber_eglas():
    name = 'E Glass Fiber'
    material = mm.isotropic_material(
        name=name,
        stiffness=72000.0,
        poisson=0.22,
        strength=None,
        density=2.54
    )
    return material


@pytest.fixture(scope='module')
def fiber_carbon_ht():
    name = 'Carbon Fiber HT'
    stiffness = dict(
        e1=230000,
        e2=13000,
        g12=50000
    )

    poisson = dict(
        nu12=0.23,
        nu23=0.3
    )

    material = mm.transverse_isotropic_material(
        name=name,
        stiffness=stiffness,
        poisson=poisson,
        strength=None,
        density=1.74
    )
    return material


@pytest.fixture(scope='module')
def fiber_carbon_hm():
    name = 'Carbon Fiber HM'
    stiffness = dict(
        e1=392000,
        e2=10000,
        g12=30000
    )

    poisson = dict(
        nu12=0.2,
        nu23=0.3
    )

    material = mm.transverse_isotropic_material(
        name=name,
        stiffness=stiffness,
        poisson=poisson,
        strength=None,
        density=1.81
    )
    return material


@pytest.fixture(scope='module')
def matrix_pa66():
    name = 'PA66'
    material = mm.isotropic_material(
        name=name,
        stiffness=2000.0,
        poisson=0.4,
        strength=65.0,
        density=1.13
    )
    return material


@pytest.fixture(scope='module')
def matrix_ep():
    name = 'Epoxy Resin'
    material = mm.isotropic_material(
        name=name,
        stiffness=3200.0,
        poisson=0.3,
        strength=90.0,
        density=1.2
    )
    return material


@pytest.fixture(scope='module')
def frp_cfk(fiber=fiber_carbon_ht(), matrix=matrix_ep(), phi=0.65):
    # set material
    strength_dict = dict(
        r_11_tensile=2000.0,
        r_11_compression=1650.0,
        r_22_tensile=70.,
        r_22_compression=240.,
        r_12=105,
    )

    material = mm.FiberReinforcedPlastic(fiber_material=fiber, matrix_material=matrix, fiber_volume_fraction=phi,
                                         name='CFRP', symmetry='mean', strength=strength_dict)

    return material


@pytest.fixture(scope='module')
def frp_gfk(fiber=fiber_eglas(), matrix=matrix_pa66(), phi=0.5):
    # set material
    material = mm.FiberReinforcedPlastic(fiber_material=fiber, matrix_material=matrix, fiber_volume_fraction=phi,
                                         name='GFRP', symmetry='mean')

    return material


@pytest.fixture(scope='module')
def fzb_lam(frp_cfk):
    lam_fzb = mm.Laminate()
    lam_fzb.add_layer(mm.Layer(material=frp_cfk, thickness=0.25, orientation=45.))
    lam_fzb.add_layer(mm.Layer(material=frp_cfk, thickness=0.25, orientation=90.))
    lam_fzb.add_layer(mm.Layer(material=frp_cfk, thickness=0.25, orientation=135.))
    lam_fzb.add_layer(mm.Layer(material=frp_cfk, thickness=0.25, orientation=0.))
    lam_fzb.add_layer(mm.Layer(material=frp_cfk, thickness=0.25, orientation=0.))
    lam_fzb.add_layer(mm.Layer(material=frp_cfk, thickness=0.25, orientation=135.))
    lam_fzb.add_layer(mm.Layer(material=frp_cfk, thickness=0.25, orientation=90.))
    lam_fzb.add_layer(mm.Layer(material=frp_cfk, thickness=0.25, orientation=45.))
    yield lam_fzb
    del lam_fzb


@pytest.fixture(scope='module')
def x_lam(frp_cfk):
    lam_x = mm.Laminate()
    lam_x.add_layer(mm.Layer(material=frp_cfk, thickness=0.25, orientation=0.))
    lam_x.add_layer(mm.Layer(material=frp_cfk, thickness=0.25, orientation=90.))
    lam_x.add_layer(mm.Layer(material=frp_cfk, thickness=0.25, orientation=0.))
    lam_x.add_layer(mm.Layer(material=frp_cfk, thickness=0.25, orientation=90.))
    lam_x.add_layer(mm.Layer(material=frp_cfk, thickness=0.25, orientation=90.))
    lam_x.add_layer(mm.Layer(material=frp_cfk, thickness=0.25, orientation=0.))
    lam_x.add_layer(mm.Layer(material=frp_cfk, thickness=0.25, orientation=90.))
    lam_x.add_layer(mm.Layer(material=frp_cfk, thickness=0.25, orientation=0.))
    yield lam_x
    del lam_x


@pytest.fixture(scope='module')
def asym_lam(m1=frp_cfk(), m2=frp_gfk(), t=0.25):
    laminate_creator = mm.StandardLaminateFactory(material=m1, layer_thickness=t)

    layers = [
        dict(orientation=0),
        dict(orientation=30),
        dict(orientation=8, material=m2),
        dict(orientation=12),
        dict(orientation=-20, thickness=0.2),
        dict(orientation=33),
        dict(orientation=90),
        dict(orientation=44, material=m2, thickness=0.5),
    ]

    lam = laminate_creator.get_laminate(layers=layers)

    return lam


@pytest.fixture(scope='module')
def puck_set_frp_cfk(mat=frp_cfk()):
    puck_set = mm.PuckStrengthSet(material=mat)
    yield puck_set
    del puck_set
