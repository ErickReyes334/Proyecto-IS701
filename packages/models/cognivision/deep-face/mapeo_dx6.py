def _n(x):
    if x is None: return 0.0
    x = float(x)
    return x/100.0 if x > 1.0 else max(0.0, min(1.0, x))

def mapear_emociones_dx6_fila(f):
    enojo    = _n(f.get('emo_enojo'))
    asco     = _n(f.get('emo_asco'))
    miedo    = _n(f.get('emo_miedo'))
    feliz    = _n(f.get('emo_feliz'))
    triste   = _n(f.get('emo_triste'))
    sorpresa = _n(f.get('emo_sorpresa'))
    neutro   = _n(f.get('emo_neutro'))

    # neutr., feliz y enojo dominantes (reglas duras)
    if neutro >= 0.80 and max(feliz, triste, miedo, enojo, asco, sorpresa) <= 0.35:
        return "Neutral"
    if feliz >= 0.75 and (triste + miedo + enojo) <= 0.55:
        return "Feliz"
    if enojo >= 0.80 and (enojo - feliz) >= 0.15 and enojo >= max(triste, miedo):
        return "Enojado"

    # asco dominante -> estrés
    if asco >= 0.80 and feliz <= 0.40:
        return "Estrés"
    if asco >= 0.60 and enojo >= 0.50 and feliz <= 0.45:
        return "Estrés"

    # combinaciones principales
    if miedo >= 0.65 and sorpresa >= 0.45 and feliz <= 0.45:
        return "Ansiedad"
    if triste >= 0.65 and feliz <= 0.35:
        return "Depresión"
    if triste >= 0.55 and miedo >= 0.45 and feliz <= 0.45:
        return "Depresión"
    if enojo >= 0.60 and miedo >= 0.50 and feliz <= 0.45:
        return "Enojado"
    if enojo >= 0.60 and asco >= 0.50 and feliz <= 0.45:
        return "Estrés"
    if feliz >= 0.60 and sorpresa >= 0.50 and (triste + miedo + enojo) <= 0.70:
        return "Feliz"

    # regla “dos negativas altas”
    neg_altas = sum(v >= 0.55 for v in [enojo, asco, miedo, triste])
    if neg_altas >= 2 and feliz <= 0.45:
        if miedo >= 0.55: 
            return "Ansiedad"
        if enojo >= 0.55 or asco >= 0.55:
            return "Estrés"
        return "Depresión"

    # puntajes (pesos ajustados para no arrastrar a Depresión con asco)
    s_feliz  = 1.15*feliz - 0.55*triste - 0.35*miedo - 0.25*enojo
    s_dep    = 1.30*triste + 0.20*miedo + 0.10*enojo - 0.60*feliz - 0.20*asco
    s_ans    = 1.30*miedo + 0.60*sorpresa - 0.30*feliz
    s_est    = 1.10*enojo + 1.00*asco + 0.50*miedo + 0.30*triste - 0.40*feliz
    s_enojo  = 1.25*enojo + 0.50*asco - 0.40*feliz
    s_neutro = 0.85*neutro - 0.45*(enojo + triste + miedo) - 0.25*feliz

    puntajes = {
        "Feliz": s_feliz, "Depresión": s_dep, "Ansiedad": s_ans,
        "Estrés": s_est, "Enojado": s_enojo, "Neutral": s_neutro
    }
    m = max(puntajes.values())
    candidatos = [k for k,v in puntajes.items() if abs(v-m) < 1e-9]

    if len(candidatos) == 1:
        return candidatos[0]
    if "Neutral" in candidatos and neutro < 0.75:
        candidatos = [c for c in candidatos if c != "Neutral"]
        
    for p in ["Enojado","Estrés","Ansiedad","Depresión","Feliz","Neutral"]:
        if p in candidatos:
            return p
