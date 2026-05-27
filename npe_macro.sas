/*===========================================================================
  NPE pipeline parametrizado
  ---------------------------------------------------------------------------
  Macro %NPE_PROCESO: ejecuta todo el flujo (target -> trazas FBE ->
  descendientes recursivos -> propagacion de flags NPE/WO) para una ventana
  de fechas concreta, escribiendo tablas de salida con sufijo por anio.

  Parametros:
    anio       Anio del target. Se usa para:
                 - leer la tabla origen r_tmp.all_target_&anio
                 - sufijar las tablas de salida (_&anio)
    fecha_ini  Fecha inicio ventana FBE/NPE. Formato literal SAS: '01dec2020'd
    fecha_fin  Fecha fin ventana.                                '31dec2021'd
    out_lib    (opcional) libreria de salida. Default = r_tmp
    max_iter   (opcional) tope de iteraciones de la recursion. Default = 50
    debug      (opcional) Y/N. Si N, borra tablas intermedias al final.

  Llamada simple:
    %NPE_PROCESO(anio=2020, fecha_ini='01dec2020'd, fecha_fin='31dec2021'd);

  Varias fechas: ver el bloque %NPE_BATCH al final.
===========================================================================*/

/* Conexion Oracle: se abre una sola vez, fuera de la macro, para no
   reconectar en cada llamada. Ajusta si prefieres abrir/cerrar por iteracion. */
libname finan4 oracle user="&user_dwh." pw="&pw_dwh." path='ADC'
        preserve_tab_names=yes connection=sharedread schema='DWH_FINANZAS';
libname wo "/risk/actual/tmp/ifrs_forward/WO_info";


%macro NPE_PROCESO(anio=, fecha_ini=, fecha_fin=,
                   out_lib=r_tmp, max_iter=50, debug=N);

    /*-- Validacion minima de parametros --------------------------------*/
    %if %length(&anio) = 0 %then %do;
        %put ERROR: NPE_PROCESO requiere al menos el parametro anio.;
        %return;
    %end;

    /*-- Si no se pasan fechas, se derivan del anio segun el patron
         estandar: 01dec(anio) -> 31dec(anio+1). Se pueden sobreescribir
         pasando fecha_ini/fecha_fin explicitas para ventanas atipicas. ----*/
    %if %length(&fecha_ini) = 0 %then
        %let fecha_ini = "01dec&anio"d;
    %if %length(&fecha_fin) = 0 %then
        %let fecha_fin = "31dec%eval(&anio + 1)"d;

    %put NOTE: ===== NPE_PROCESO anio=&anio  ventana &fecha_ini - &fecha_fin =====;

    /*-- Nombres de tablas, todos derivados de &anio --------------------*/
    %local in_target cet cet_desc cet_final cet_final2;
    %let in_target  = r_tmp.all_target_&anio;            /* entrada      */
    %let cet        = &out_lib..all_target_&anio._cet;
    %let cet_desc   = &out_lib..all_target_&anio._cet_desc;
    %let cet_final  = &out_lib..all_target_&anio._cet_final;
    %let cet_final2 = &out_lib..all_target_&anio._cet_final2;

    /*-------------------------------------------------------------------
      Paso 1: base de contratos objetivo
    -------------------------------------------------------------------*/
    data &cet;
        format obs_fch ddmmyy10.;
        set &in_target(where=(society = 4));
        contract = put(account_id, $14.);
        obs_fch  = intnx('month',
                         mdy(mod(observation_date,100), 1, int(observation_date/100)),
                         0, 'e');
    run;

    /*-------------------------------------------------------------------
      Paso 2: trazas FBE en la ventana
    -------------------------------------------------------------------*/
    proc sql;
        create table work.fbe_trans as
        select distinct t.contrato_origen, t.contrato_fbe, t.fch_fbe,
               t.tipo, t.imp_origen, t.IMP_FINAN
        from finan4.FBE_ND_H as t
        inner join (select distinct contract from &cet) as c
            on t.contrato_origen = c.contract
        where t.fch_fbe >  &fecha_ini.
          and t.fch_fbe <= &fecha_fin.
          and t.tipo not in ('C','A');
    quit;

    /*-------------------------------------------------------------------
      Paso 3: descendientes recursivos con hash objects
    -------------------------------------------------------------------*/
    proc sql;
        create table work.desc_acum as
        select distinct
               contrato_origen as raiz       length=14,
               contrato_origen as cto_actual length=14,
               .  as fch_fbe     format=datetime20.,
               .  as fch_fbe_eom format=ddmmyy10.,
               "" as tipo        length=1,
               0  as nivel
        from work.fbe_trans;
    quit;

    data work.frontier;
        set work.desc_acum;
    run;

    %local iter nuevos;
    %let iter   = 0;
    %let nuevos = 1;

    %do %while (&nuevos ne 0 and &iter < &max_iter);

        %let iter = %eval(&iter + 1);

        data work._gen&iter (drop=_rc found _padre contrato_origen contrato_fbe);
            length raiz cto_actual $14 tipo $1;
            format fch_fbe datetime20. fch_fbe_eom ddmmyy10.;

            if _n_ = 1 then do;
                declare hash h_edges(dataset:'work.fbe_trans', multidata:'yes');
                h_edges.defineKey('contrato_origen');
                h_edges.defineData('contrato_fbe','fch_fbe','tipo');
                h_edges.defineDone();

                declare hash h_vis(dataset:'work.desc_acum');
                h_vis.defineKey('raiz','cto_actual');
                h_vis.defineDone();
                call missing(contrato_fbe, contrato_origen);
            end;

            set work.frontier(rename=(cto_actual=_padre) keep=raiz cto_actual nivel);
            contrato_origen = _padre;

            _rc = h_edges.find();
            do while (_rc = 0);
                cto_actual = contrato_fbe;
                found = (h_vis.check() = 0);
                if not found then do;
                    h_vis.add();
                    nivel       = &iter;
                    fch_fbe_eom = intnx('month', datepart(fch_fbe), 0, 'e');
                    output;
                end;
                _rc = h_edges.find_next();
            end;
        run;

        proc sql noprint;
            select count(*) into :nuevos trimmed from work._gen&iter;
        quit;

        %if &nuevos ne 0 %then %do;
            proc append base=work.desc_acum data=work._gen&iter
                        (keep=raiz cto_actual fch_fbe fch_fbe_eom tipo nivel)
                        force;
            run;
            data work.frontier;
                set work._gen&iter(keep=raiz cto_actual nivel);
            run;
        %end;

    %end;

    %put NOTE: anio=&anio recursion terminada en &iter iteraciones.;

    proc datasets library=work nolist; delete _gen: frontier; quit;

    proc sort data=work.desc_acum; by raiz fch_fbe_eom fch_fbe; run;

    data work.desc_acum_final;
        set work.desc_acum;
        by raiz fch_fbe_eom;
        if last.fch_fbe_eom then output;
    run;

    /*-------------------------------------------------------------------
      Paso 4: join base + descendientes (hash lookup)
    -------------------------------------------------------------------*/
    data &cet_desc;
        if _n_ = 1 then do;
            declare hash h(dataset:'work.desc_acum_final');
            h.defineKey('raiz','fch_fbe_eom');
            h.defineData('cto_actual','fch_fbe','tipo');
            h.defineDone();
            call missing(cto_actual, fch_fbe, tipo);
        end;

        set &cet;
        length cto_actual $14 tipo_fbe $1;
        format fch_fbe datetime20.;

        raiz        = contract;
        fch_fbe_eom = obs_fch;

        if h.find() ne 0 then call missing(cto_actual, fch_fbe, tipo);
        tipo_fbe = tipo;
        drop raiz fch_fbe_eom tipo;
    run;

    /*-------------------------------------------------------------------
      Paso 5: forward fill del ultimo cto_actual
    -------------------------------------------------------------------*/
    proc sort data=&cet_desc; by contract obs_fch; run;

    data &cet_final;
        set &cet_desc;
        by contract;
        length _ultimo_cto $14;
        retain _ultimo_cto;

        if first.contract then _ultimo_cto = "";
        if not missing(cto_actual) then _ultimo_cto = cto_actual;

        if not missing(_ultimo_cto) then cto_actual = _ultimo_cto;
        else cto_actual = contract;

        drop _ultimo_cto;
    run;

    /*-------------------------------------------------------------------
      Paso 6: flag NPE de descendientes
    -------------------------------------------------------------------*/
    proc sql;
        create table work.target_aux as
        select c.cto_actual as contract,
               t.observation_date
        from (select distinct cto_actual from &cet_final) as c
        inner join (
            select contract,
                   datepart(date_base) as observation_date format=ddmmyy10.
            from r_nd.ND_OUTPUT
            where date_base >= &fecha_ini. and date_base <= &fecha_fin.
              and flg_npe = '1'
        ) as t
            on c.cto_actual = t.contract;
    quit;

    data &cet_final;
        if _n_ = 1 then do;
            declare hash hn();
            hn.defineKey('k1','k2');
            hn.defineDone();
            do until (eof1);
                set work.target_aux(rename=(contract=k1 observation_date=k2)) end=eof1;
                hn.replace();
            end;
        end;

        set &cet_final;
        k1 = cto_actual;
        k2 = obs_fch;
        flg_npe_desc = (hn.check() = 0);
        drop k1 k2;
    run;

    /*-------------------------------------------------------------------
      Paso 7: flag write-off (WO)
    -------------------------------------------------------------------*/
    data &cet_final;
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

        set &cet_final;
        k1 = contract;
        k2 = observation_date;
        flg_npe_wo = (hw.check() = 0);
        drop k1 k2;
    run;

    /*-------------------------------------------------------------------
      Paso 8: propagacion WO + total
    -------------------------------------------------------------------*/
    proc sort data=&cet_final; by contract obs_fch; run;

    data &cet_final2;
        set &cet_final;
        by contract;
        retain _wo;

        if first.contract then _wo = 0;
        if flg_npe_wo = 1 then _wo = 1;
        flg_npe_wo = _wo;

        flag_npe_total = sum(0, flag_npe, flg_npe_wo, flg_npe_desc);

        drop _wo;
    run;

    /*-- Limpieza de intermedios (salvo modo debug) ---------------------*/
    %if %upcase(&debug) ne Y %then %do;
        proc datasets library=work nolist;
            delete fbe_trans desc_acum desc_acum_final target_aux;
        quit;
    %end;

    %put NOTE: ===== NPE_PROCESO anio=&anio COMPLETADO -> &cet_final2 =====;

%mend NPE_PROCESO;


/*===========================================================================
  Opcion A: llamadas individuales.
  Con el patron estandar (dic anio -> dic anio+1) basta con pasar el anio;
  las fechas se derivan solas. Pasa fecha_ini/fecha_fin solo si la ventana
  de ese anio es atipica.
===========================================================================*/
%NPE_PROCESO(anio=2020);
/*
%NPE_PROCESO(anio=2021);
%NPE_PROCESO(anio=2022);

  Ejemplo con ventana atipica (override explicito):
%NPE_PROCESO(anio=2023, fecha_ini='01jan2023'd, fecha_fin='30jun2024'd);
*/


/*===========================================================================
  Opcion B: batch sobre una lista de anios. Como las fechas se derivan del
  anio, basta con enumerar los anios separados por espacios.
  Para ventanas atipicas usa llamadas individuales con override (Opcion A).
===========================================================================*/
%macro NPE_BATCH(anios=);
    %local i a;
    %let i = 1;
    %let a = %scan(&anios, &i, %str( ));
    %do %while (%length(&a) > 0);
        %NPE_PROCESO(anio=&a);
        %let i = %eval(&i + 1);
        %let a = %scan(&anios, &i, %str( ));
    %end;
%mend NPE_BATCH;

/* Ejemplo: procesa 2020, 2021 y 2022 de forma independiente */
/*
%NPE_BATCH(anios=2020 2021 2022);
*/
