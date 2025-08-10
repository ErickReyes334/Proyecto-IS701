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

    # reglas combinadas de detección rápida
    if neutro >= 0.75 and max(feliz, triste, miedo, enojo, asco, sorpresa) <= 0.35:
        return "Neutral"

    # miedo + sorpresa => ansiedad
    if miedo >= 0.50 and sorpresa >= 0.45 and feliz <= 0.40:
        return "Ansiedad"

    # enojo + asco => estrés
    if enojo >= 0.50 and asco >= 0.45 and feliz <= 0.40:
        return "Estrés"

    # tristeza + miedo => depresión con componente ansiosa
    if triste >= 0.50 and miedo >= 0.45 and feliz <= 0.40:
        return "Depresión"

    # enojo + miedo => enojo reactivo (ira + temor)
    if enojo >= 0.50 and miedo >= 0.50:
        return "Enojado"

    # felicidad + sorpresa => feliz excitado (euforia)
    if feliz >= 0.55 and sorpresa >= 0.50:
        return "Feliz"

    # tristeza + asco => rechazo + depresión
    if triste >= 0.55 and asco >= 0.40:
        return "Depresión"

    #  reglas de detección individual
    if feliz  >= 0.60 and (triste + miedo) <= 0.35: return "Feliz"
    if triste >= 0.65 and feliz <= 0.35:            return "Depresión"
    if miedo  >= 0.60 and feliz <= 0.45:            return "Ansiedad"
    if enojo  >= 0.60 and feliz <= 0.45:            return "Enojado"

    # sistema de puntajes
    s_feliz  = 1.1*feliz - 0.5*triste - 0.3*miedo
    s_dep    = 1.2*triste + 0.4*asco + 0.3*enojo - 0.5*feliz
    s_ans    = 1.1*miedo + 0.5*sorpresa - 0.3*feliz
    s_est    = 0.9*enojo + 0.8*asco + 0.5*miedo + 0.3*triste - 0.3*feliz
    s_enojo  = 1.2*enojo + 0.6*asco - 0.3*feliz
    s_neutro = 0.8*neutro - 0.4*(enojo + triste + miedo) - 0.2*feliz

    puntajes = {
        "Feliz": s_feliz, "Depresión": s_dep, "Ansiedad": s_ans,
        "Estrés": s_est, "Enojado": s_enojo, "Neutral": s_neutro
    }
    m = max(puntajes.values())
    candidatos = [k for k,v in puntajes.items() if abs(v-m) < 1e-9]

    if len(candidatos) == 1:
        return candidatos[0]
    if "Neutral" in candidatos and neutro < 0.70:
        candidatos = [c for c in candidatos if c != "Neutral"]
    for p in ["Feliz","Ansiedad","Estrés","Enojado","Depresión","Neutral"]:
        if p in candidatos: return p
