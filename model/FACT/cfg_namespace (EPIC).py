from types import SimpleNamespace

BU = SimpleNamespace(
    a="sa",
    a_dim=None,
    a_ffdim=None,
    a_layers=1,
    a_nhead=8,
    dropout=None,
    f=None,
    f_dim=None,
    f_layers=3,
    f_ln=None,
    f_ngp=None,
    hid_dim=None,
    s_layers=1
)

Bi = SimpleNamespace(
    a="sca",
    a_dim=128,
    a_ffdim=128,
    a_layers=2,
    a_nhead=8,
    dropout=0.1,
    f="m2",
    f_dim=128,
    f_layers=3,
    f_ln=False,
    f_ngp=1,
    hid_dim=128
)

Bu = SimpleNamespace(
    a="sa",
    a_dim=None,
    a_ffdim=None,
    a_layers=1,
    a_nhead=8,
    dropout=None,
    f=None,
    f_dim=None,
    f_layers=1,
    f_ln=None,
    f_ngp=None,
    hid_dim=None
)

FACT = SimpleNamespace(
    block="iuU",
    cmr=0.3,
    # fpos=False,
    fpos = True,
    ntoken=150,
    trans=False,
    mwt=0.75,
)

Loss = SimpleNamespace(
    a2fc=1.0,
    bgw=1.0,
    match="o2o",
    nullw=-1.0,
    pc=0.2,
    sw=5.0,
)

TM = SimpleNamespace(
    inplace=True,
    m=5,
    p=0.05,
    t=30,
    use=True
)

aux = SimpleNamespace(
    debug=False,
    eval_every=2000,
    gpu=0,
    mark="",
    print_every=1000,
    resume="max",
    runid=0
)

# Create the main namespace to hold all the other namespaces
cfg = SimpleNamespace(
    BU=BU,
    Bi=Bi,
    Bu=Bu,
    FACT=FACT,
    Loss=Loss,
    TM=TM,
    aux=aux
)