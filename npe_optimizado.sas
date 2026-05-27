/*===========================================================================
  NPE pipeline - versión optimizada
  Cambios clave:
    1. Recursión de descendientes con hash objects (evita el O(n^2) del
       'where not exists' + proc append re-escaneando desc_acum cada vuelta).
    2. Joins finales con hash lookup en DATA step (evita sort-merge en disco
       cuando la tabla de lookup cabe en memoria).
    3. Fusión de los pasos de propagación con retain en menos DATA steps.
    4. Trabajo intermedio en WORK; solo se materializa en r_tmp lo necesario.
  Revisar tamaños de tablas de lookup antes de subir a producción: si alguna
  no cabe en memoria, volver al merge clásico para ese paso concreto.
===========================================================================*/

/*---------------------------------------------------------------------------
  Paso 1: base de contratos objetivo (sin cambios sustanciales)
---------------------------------------------------------------------------*/
data r_tmp.all_target_2020_cet;
    format obs_fch ddmmyy10.;
    set r_tmp.all_target_2020(where=(society = 4));
    contract = put(account_id, $14.);
    obs_fch  = intnx('month',
                     mdy(mod(observation_date,100), 1, int(observation_date/100)),
                     0, 'e');
run;

libname finan4 oracle user="&user_dwh." pw="&pw_dwh." path='ADC'
        preserve_tab_names=yes connection=sharedread schema='DWH_FINANZAS';

%let fecha_ini='01dec2020'd;
%let fecha_fin='31dec2021'd;

/*---------------------------------------------------------------------------
  Paso 2: trazas FBE. El 'in (select distinct contract...)' se sustituye por
  un inner join contra la lista de contratos, que el optimizador de Oracle
  puede empujar (predicate pushdown) mejor que un IN correlacionado.
  Mantener distinct solo si FBE_ND_H puede traer duplicados reales.
---------------------------------------------------------------------------*/
proc sql;
    create table fbe_trans as
    select distinct t.contrato_origen, t.contrato_fbe, t.fch_fbe,
           t.tipo, t.imp_origen, t.IMP_FINAN
    from finan4.FBE_ND_H as t
    inner join (select distinct contract from r_tmp.all_target_2020_cet) as c
        on t.contrato_origen = c.contract
    where t.fch_fbe > &fecha_ini.
      and t.fch_fbe <= &fecha_fin.
      and t.tipo not in ('C','A');
quit;

/*---------------------------------------------------------------------------
  Paso 3: descendientes recursivos con hash objects.

  - h_edges: indexa fbe_trans por contrato_origen, multidata, para iterar
    todos los hijos de un contrato dado.
  - h_visited: set de pares (raiz, cto) ya alcanzados -> anti-join en memoria.
  - Se procesa por generaciones leyendo la generación anterior desde un
    dataset 'frontier' y escribiendo la nueva, en vez de re-escanear todo.
---------------------------------------------------------------------------*/
%macro OBTENER_DESCENDIENTES_HASH();

    /* nivel 0: cada origen es su propia raiz */
    proc sql;
        create table desc_acum as
        select distinct
               contrato_origen as raiz       length=14,
               contrato_origen as cto_actual length=14,
               .  as fch_fbe     format=datetime20.,
               .  as fch_fbe_eom format=ddmmyy10.,
               "" as tipo        length=1,
               0  as nivel
        from fbe_trans;
    quit;

    /* la frontera inicial es el nivel 0 */
    data frontier;
        set desc_acum;
    run;

    %let iter   = 0;
    %let nuevos = 1;

    %do %while (&nuevos ne 0 and &iter < 50);

        %let iter = %eval(&iter + 1);

        data _gen&iter (drop=_rc found);
            length raiz cto_actual $14 tipo $1;
            format fch_fbe datetime20. fch_fbe_eom ddmmyy10.;

            /* hash de aristas: contrato_origen -> (hijo, fecha, tipo) */
            if _n_ = 1 then do;
                declare hash h_edges(dataset:'fbe_trans', multidata:'yes');
                h_edges.defineKey('contrato_origen');
                h_edges.defineData('contrato_fbe','fch_fbe','tipo');
                h_edges.defineDone();

                /* hash de visitados, cargado con TODO desc_acum hasta ahora */
                declare hash h_vis(dataset:'desc_acum');
                h_vis.defineKey('raiz','cto_actual');
                h_vis.defineDone();
                call missing(contrato_fbe, contrato_origen);
            end;

            /* recorre la frontera (generación anterior) */
            set frontier(rename=(cto_actual=_padre) keep=raiz cto_actual nivel);
            contrato_origen = _padre;

            /* itera todos los hijos del padre */
            _rc = h_edges.find();
            do while (_rc = 0);
                /* ¿ya visitado (raiz, hijo)? */
                cto_actual = contrato_fbe;
                found = (h_vis.check() = 0);
                if not found then do;
                    h_vis.add();                       /* marca visitado */
                    nivel       = &iter;
                    fch_fbe_eom = intnx('month', datepart(fch_fbe), 0, 'e');
                    output;
                end;
                _rc = h_edges.find_next();
            end;
        run;

        proc sql noprint;
            select count(*) into :nuevos trimmed from _gen&iter;
        quit;

        %if &nuevos ne 0 %then %do;
            proc append base=desc_acum data=_gen&iter
                        (keep=raiz cto_actual fch_fbe fch_fbe_eom tipo nivel)
                        force;
            run;
            /* la nueva frontera es la generación recién creada */
            data frontier;
                set _gen&iter(keep=raiz cto_actual nivel);
            run;
        %end;

    %end;

    proc datasets library=work nolist; delete _gen: frontier; quit;

    /* último FBE por raiz y mes (last.fch_fbe_eom). Un solo sort + by */
    proc sort data=desc_acum; by raiz fch_fbe_eom fch_fbe; run;

    data desc_acum_final;
        set desc_acum;
        by raiz fch_fbe_eom;
        if last.fch_fbe_eom then output;
    run;
%mend;
%OBTENER_DESCENDIENTES_HASH()

/*---------------------------------------------------------------------------
  Paso 4: join base + descendientes.
  desc_acum_final suele ser pequeña -> hash lookup por (raiz, fch_fbe_eom),
  evitando ordenar all_target por esa clave.
---------------------------------------------------------------------------*/
data r_tmp.all_target_2020_CET_DESC;
    if _n_ = 1 then do;
        declare hash h(dataset:'desc_acum_final');
        h.defineKey('raiz','fch_fbe_eom');
        h.defineData('cto_actual','fch_fbe','tipo');
        h.defineDone();
        call missing(cto_actual, fch_fbe, tipo);
    end;

    set r_tmp.all_target_2020_CET;
    length cto_actual $14 tipo_fbe $1;
    format fch_fbe datetime20.;

    /* claves del lookup deben llamarse igual que en el hash */
    raiz        = contract;
    fch_fbe_eom = obs_fch;

    if h.find() ne 0 then call missing(cto_actual, fch_fbe, tipo);
    tipo_fbe = tipo;
    drop raiz fch_fbe_eom tipo;
run;

/*---------------------------------------------------------------------------
  Paso 5: arrastre del último cto_actual (forward fill). Necesita orden por
  contract obs_fch. Un solo sort sirve para este paso y los siguientes.
---------------------------------------------------------------------------*/
proc sort data=r_tmp.all_target_2020_CET_DESC; by contract obs_fch; run;

data r_tmp.all_target_2020_CET_final;
    set r_tmp.all_target_2020_CET_DESC;
    by contract;
    length _ultimo_cto $14;
    retain _ultimo_cto;

    if first.contract then _ultimo_cto = "";
    if not missing(cto_actual) then _ultimo_cto = cto_actual;

    if not missing(_ultimo_cto) then cto_actual = _ultimo_cto;
    else cto_actual = contract;

    drop _ultimo_cto;
run;

/*---------------------------------------------------------------------------
  Paso 6: flag NPE de descendientes (target_aux).
  Solo se necesita la existencia del par (cto_actual, observation_date),
  así que un hash de claves (sin data) basta para el flag 0/1.
---------------------------------------------------------------------------*/
proc sql;
    create table target_aux as
    select c.cto_actual as contract,
           t.observation_date
    from (select distinct cto_actual from r_tmp.all_target_2020_CET_final) as c
    inner join (
        select contract,
               datepart(date_base) as observation_date format=ddmmyy10.
        from r_nd.ND_OUTPUT
        where date_base >= &fecha_ini. and date_base <= &fecha_fin.
          and flg_npe = '1'
    ) as t
        on c.cto_actual = t.contract;
quit;

/* Lookup de existencia por (cto_actual, obs_fch). Se cargan las claves de
   target_aux en variables temporales k1/k2 para no colisionar con columnas
   reales del dataset principal. */
data r_tmp.all_target_2020_CET_final;
    if _n_ = 1 then do;
        declare hash hn();
        hn.defineKey('k1','k2');
        hn.defineDone();
        /* carga manual de las claves de target_aux */
        do until (eof1);
            set target_aux(rename=(contract=k1 observation_date=k2)) end=eof1;
            hn.replace();
        end;
    end;

    set r_tmp.all_target_2020_CET_final;
    k1 = cto_actual;
    k2 = obs_fch;
    flg_npe_desc = (hn.check() = 0);
    drop k1 k2;
run;

/*---------------------------------------------------------------------------
  Paso 7: flag write-off (WO). Mismo patrón de hash de existencia.
---------------------------------------------------------------------------*/
libname wo "/risk/actual/tmp/ifrs_forward/WO_info";

data r_tmp.all_target_2020_CET_final;
    if _n_ = 1 then do;
        declare hash hw();
        hw.defineKey('k1','k2');
        hw.defineDone();
        do until (eof1);
            set wo.wo_prectx_4_tot(where=(NATURE=392)
                                   rename=(contrato=k1 mes=k2)) end=eof1;
            hw.replace();
        end;
    end;

    set r_tmp.all_target_2020_CET_final;
    k1 = contract;
    k2 = observation_date;
    flg_npe_wo = (hw.check() = 0);
    drop k1 k2;
run;

/*---------------------------------------------------------------------------
  Paso 8: propagación del WO (forward fill) + total, fusionados en un solo
  DATA step. Requiere orden por contract obs_fch.
  OJO: el original usa 'flag_npe' (con A) en el sum y 'flg_npe' en el resto.
  Mantengo 'flag_npe' tal cual estaba; verificar que esa variable existe.
---------------------------------------------------------------------------*/
proc sort data=r_tmp.all_target_2020_CET_final; by contract obs_fch; run;

data r_tmp.all_target_2020_CET_final2;
    set r_tmp.all_target_2020_CET_final;
    by contract;
    retain _wo;

    if first.contract then _wo = 0;
    if flg_npe_wo = 1 then _wo = 1;
    flg_npe_wo = _wo;

    flag_npe_total = sum(0, flag_npe, flg_npe_wo, flg_npe_desc);

    drop _wo;
run;
