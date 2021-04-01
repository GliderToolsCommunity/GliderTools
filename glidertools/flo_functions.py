#!/usr/bin/env python
"""
@package ion_functions.data.flo_functions
@file ion_functions/data/flo_functions.py
@author Christopher Wingard, Craig Risien, Russell Desiderio
@brief Module containing Fluorometer Three Wavelength (FLORT) and Fluorometer
    Two Wavelength (FLORD) instrument family related functions

Copyright (c) 2010, 2011 The Regents of the University of California

Permission to use, copy, modify, and distribute this software and its
documentation for educational, research and non-profit purposes, without fee,
and without a written agreement is hereby granted, provided that the above
copyright notice, this paragraph and the following three paragraphs appear
in all copies.

Permission to make commercial use of this software may be obtained
by contacting:
Technology Transfer Office
9500 Gilman Drive, Mail Code 0910
University of California
La Jolla, CA 92093-0910
(858) 534-5815
invent@ucsd.edu

THIS SOFTWARE IS PROVIDED BY THE REGENTS OF THE UNIVERSITY OF CALIFORNIA AND
CONTRIBUTORS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import absolute_import as _ai
from __future__ import print_function as _pf
from __future__ import unicode_literals as _ul

import numexpr as ne
import numpy as np


def flo_bback_total(beta, degC, psu, theta, wlngth, xfactor):
    """
    Description:
        This function calculates the OOI Level 2 Optical Backscatter data product
        (FLUBSCT_L2), which is calculated using data from the WET Labs, Inc. ECO
        fluorometer family of instruments (FLORD, FLORT, FLNTU) at the wavelength
        specified by the wlngth argument. See Notes.
    Implemented by:
        2013-07-16: Christopher Wingard. Initial Code.
        2014-04-23: Christopher Wingard. Slight revisions to address
                    integration issues and to meet intent of DPS.
        2015-10-26: Russell Desiderio. Deleted default values in argument list.
                                       Revised documentation. Added Notes section.
    Usage:
        bback = flo_bback_total(beta, degC, psu, theta, wlngth, xfactor)
            where
        bback = total (seawater + particulate) optical backscatter coefficient
            at wavelength wlngth (FLUBSCT_L2) [m-1].
        beta = value of the volume scattering function (seawater + particulate) measured
            at angle theta and at wavelength wlngth (FLUBSCT_L1) [m-1 sr-1].
        degC = in situ water temperature from co-located CTD [deg_C].
        psu = in situ salinity from co-located CTD [psu].
        theta = effective (centroid) optical backscatter scattering angle [degrees] which
            is a function of the sensor geometry of the measuring instrument. See Notes.
        wlngth = optical backscatter measurement wavelength [nm]. See Notes.
        xfactor = X (Chi) factor which scales the particulate scattering value at a particular
            backwards angle to the total particulate backscattering coefficient integrated
            over all backwards angles. See Notes.
    Notes:
        The values to be used for theta, chi factor, and wavelength depend on the instrument.
        For WETLabs 'ECO' instruments with 3 optical channels (FLORT D,J,K,M,N,O; FLORD D) the
        centroid angle theta is 124 degrees (not 117 as in WETLabs' older documentation) and the
        appropriate chi factor for these instruments is 1.076 [Sullivan, Twardowski, Zaneveld, and
        Moore, 2013, Table 6.2b, "ECO-BB" (= ECO-BB3)]. The ECO-BB3 was initially mis-classified
        as an OPTAA series M instrument, and is now classified as a FLORT series O instrument.
        For 'ECO' instruments with 2 optical channels (WETLabs models FLBB and FLNTU: FLORD G,L,M
        and the FLNTU component of FLORT A which is now designated as FLNTU series A) the centroid
        angle theta is 140 degrees and the chi factor is 1.096 (Mike Twardowski, personal
        communication).
        All optical backscatter channels of FLORD and FLORT instruments use a light source at
        a nominal wavelength of 700nm. All three optical channels of an ECO-BB3 are backscatter
        channels, typically at 3 different wavelengths in the visible (blue, green and red). The
        wavelength dependence of the chi factor is a subject of current research. At this time
        it is thought to be very weakly dependent on wavelength, if at all, and so the above chi
        factors should be used for scattering calculations involving any visible wavelength
        (Mike Twardowski, personal communication).
        The chi factor is a function of angle and the sensor geometry of the instrument used to
        measure the volume scattering function at a given backwards scattering angle. It is a
        scaling factor relating the particulate scattering at that angle to the particulate
        total backscatter coefficient (the latter is the integral over all backwards angles of the
        volume scattering function due to particles). The chi factor is not an "angular resolution"
        as it has been labelled in the OOI program.
        Depending on context within the documentation the word 'total' can have several meanings:
            (1) seawater + particulate scattering
            (2) forward + backward scattering
            (3) backscatter integrated over all backwards wavelengths.
    References:
        OOI (2012). Data Product Specification for Optical Backscatter (Red
            Wavelengths). Document Control Number 1341-00540 (version 1-05).
            https://alfresco.oceanobservatories.org/ (See: Company Home >>
            OOI >> Controlled >> 1000 System Level >>
            1341-00540_Data_Product_SPEC_FLUBSCT_OOI.pdf)
        Sullivan, J.M., M.S. Twardowski, J.R.V. Zaneveld, and C.C. Moore. Measuring optical
            backscattering in water. Chapter 6 in Light Scattering Reviews 7: Radiative Transfer
            and Optical Properties of Atmosphere and Underlying Surface (2013) pp 189-224.
    """
    # calculate:
    #    betasw, the theoretical value of the volume scattering function for seawater only
    #        at the measurement angle theta and wavelength wlngth [m-1 sr-1], and,
    #    bsw, the theoretical value for the total (in this case meaning forward + backward)
    #         scattering coefficient for seawater (also with no particulate contribution)
    #         at wavelength wlngth [m-1].
    # Values below are computed using provided code from Zhang et al 2009.
    betasw, bsw = flo_zhang_scatter_coeffs(degC, psu, theta, wlngth)

    # calculate the volume scattering at angle theta of particles only, betap.
    #     beta = scattering measured at angle theta for seawater + particulates
    #     betasw = theoretical seawater only value calculated at angle theta
    betap = beta - betasw

    # calculate the particulate backscatter coefficient bbackp [m-1] which is effectively
    # the particulate scattering function integrated over all backwards angles. The factor
    # of 2*pi arises from the integration over the (implicit) polar angle variable.
    pi = np.pi
    bbackp = xfactor * 2.0 * pi * betap

    # calculate the backscatter coefficient due to seawater from the total (forward + backward)
    # scattering coefficient bsw. because the effective scattering centers in pure seawater are
    # much smaller than the wavelength, the shape of the scattering function is symmetrical in
    # the forward and backward directions.
    bbsw = bsw / 2

    # calculate the total (particulates + seawater) backscatter coefficient.
    bback = bbackp + bbsw

    return bback


def flo_scat_seawater(degC, psu, theta, wlngth, delta=0.039):
    """
    Description:
        Computes the scattering coefficient of seawater based on the
        computation of Zhang et al 2009 as presented in the DPS for Optical
        Backscatter (red wavelengths).
    Implemented by:
        2014-04-24: Christopher Wingard. Initial Code
    Usage:
        bsw = flo_scat_seawater(degC, psu, theta, wlngth, delta)
            where
        bsw = total scattering coefficient of pure seawater [m-1]
        degC = in situ water temperature from co-located CTD [deg_C]
        psu = in situ salinity from co-located CTD [psu]
        theta = optical backscatter angle [degrees].
            See Notes to function flo_bback_total.
        wlngth = optical backscatter measurement wavelength [nm].
            See Notes to function flo_bback_total.
        delta = depolarization ratio [unitless]. Default of 0.039 is assumed.
    References:
        OOI (2012). Data Product Specification for Optical Backscatter (Red
            Wavelengths). Document Control Number 1341-00540 (version 1-05).
            https://alfresco.oceanobservatories.org/ (See: Company Home >>
            OOI >> Controlled >> 1000 System Level >>
            1341-00540_Data_Product_SPEC_FLUBSCT_OOI.pdf)
    """
    _, bsw = flo_zhang_scatter_coeffs(degC, psu, theta, wlngth, delta)
    return bsw


def flo_zhang_scatter_coeffs(degC, psu, theta, wlngth, delta=0.039):
    """
    Description:
        Computes scattering coefficients for seawater (both the volume scattering at
        a given angle theta and the total scattering coefficient integrated over all
        scattering angles) at a given wavelength wlngth based on the computation of
        Zhang et al 2009 as presented in the DPS for Optical Backscatter.
        This code is derived from Matlab code developed and made available
        online by:
            Dr. Xiaodong Zhang
            Associate Professor
            Department of Earth Systems Science and Policy
            University of North Dakota
            http://www.und.edu/instruct/zhang/
    Implemented by:
        2013-07-15: Christopher Wingard. Initial Code
    Usage:
        betasw, bsw = flo_zhang_scatter_coeffs(degC, psu, theta, wlngth, delta)
            where
        betasw = value for the volume scattering function of pure seawater
            at angle theta and wavelength wlngth [m-1 sr-1]
        bsw = total scattering coefficient of pure seawater [m-1]
        degC = in situ water temperature from co-located CTD [deg_C]
        psu = in situ salinity from co-located CTD [psu]
        theta = optical backscatter angle [degrees].
            See Notes to function flo_bback_total.
        wlngth = optical backscatter measurement wavelength [nm].
            See Notes to function flo_bback_total.
        delta = depolarization ratio [unitless]. Default of 0.039 is assumed.
    References:
        OOI (2012). Data Product Specification for Optical Backscatter (Red
            Wavelengths). Document Control Number 1341-00540 (version 1-05).
            https://alfresco.oceanobservatories.org/ (See: Company Home >>
            OOI >> Controlled >> 1000 System Level >>
            1341-00540_Data_Product_SPEC_FLUBSCT_OOI.pdf)
    """
    # values of the constants
    Na = 6.0221417930e23  # Avogadro's constant
    Kbz = 1.3806503e-23  # Boltzmann constant
    degK = degC + 273.15  # Absolute temperature
    M0 = 0.018  # Molecular weight of water in kg/mol
    pi = np.pi

    # convert the scattering angle from degrees to radians
    rad = np.radians(theta)

    # calculate the absolute refractive index of seawater and the partial
    # derivative of seawater refractive index with regards to salinity.
    nsw, dnds = flo_refractive_index(wlngth, degC, psu)

    # isothermal compressibility is from Lepple & Millero (1971,Deep
    # Sea-Research), pages 10-11 The error ~ +/-0.004e-6 bar^-1
    icomp = flo_isotherm_compress(degC, psu)

    # density of seawater from UNESCO 38 (1981).
    rho = flo_density_seawater(degC, psu)

    # water activity data of seawater is from Millero and Leung (1976, American
    # Journal of Science, 276, 1035-1077). Table 19 was reproduced using
    # Eq.(14,22,23,88,107) that were fitted to polynominal equation. dlnawds is
    # a partial derivative of the natural logarithm of water activity with
    # regards to salinity.
    dlnawds = ne.evaluate(
        "(-5.58651e-4 + 2.40452e-7 * degC - 3.12165e-9 * degC**2 + 2.40808e-11 * degC**3) +"
        "1.5 * (1.79613e-5 - 9.9422e-8 * degC + 2.08919e-9 * degC**2 - 1.39872e-11 * degC**3) *"
        "psu**0.5 + 2 * (-2.31065e-6 - 1.37674e-9 * degC - 1.93316e-11 * degC**2) * psu"
    )

    # density derivative of refractive index from PMH model
    dfri = ne.evaluate(
        "(nsw**2 - 1.0) * (1.0 + 2.0/3.0 * (nsw**2 + 2.0)"
        "* (nsw/3.0 - 1.0/3.0 / nsw)**2)"
    )

    # volume scattering at 90 degrees due to the density fluctuation
    beta_df = ne.evaluate(
        "pi**2 / 2.0 * (wlngth*1e-9)**-4 * Kbz * degK * icomp "
        "* dfri**2 * (6.0 + 6.0 * delta) / (6.0 - 7.0 * delta)"
    )

    # volume scattering at 90 degree due to the concentration fluctuation
    flu_con = ne.evaluate("psu * M0 * dnds**2 / rho / -dlnawds / Na")
    beta_cf = ne.evaluate(
        "2.0 * pi**2 * (wlngth * 1e-9)**-4 * nsw**2 * flu_con"
        "* (6.0 + 6.0 * delta) / (6.0 - 7.0 * delta)"
    )

    # total volume scattering at 90 degree
    beta90sw = beta_df + beta_cf

    # total scattering coefficient of seawater (m-1)
    bsw = ne.evaluate("8.0 * pi / 3.0 * beta90sw * ((2.0 + delta) / (1.0 + delta))")

    # total volume scattering coefficient of seawater (m-1 sr-1)
    betasw = ne.evaluate(
        "beta90sw * (1.0 + ((1.0 - delta) / (1.0 + delta)) * cos(rad)**2)"
    )

    return betasw, bsw


def flo_refractive_index(wlngth, degC, psu):
    """
    Helper function for flo_zhang_scatter_coeffs
    @param wlngth backscatter measurement wavlength (nm)
    @param degC in situ water temperature (deg_C)
    @param psu in site practical salinity (psu)
    @retval nsw absolute refractive index of seawater
    @retval dnds partial derivative of seawater refractive index with regards to
        seawater.
    """
    # refractive index of air is from Ciddor (1996, Applied Optics).
    n_air = ne.evaluate(
        "1.0 + (5792105.0 / (238.0185 - 1 / (wlngth/1e3)**2)"
        "+ 167917.0 / (57.362 - 1 / (wlngth/1e3)**2)) / 1e8"
    )

    # refractive index of seawater is from Quan and Fry (1994, Applied Optics)
    n0 = 1.31405
    n1 = 1.779e-4
    n2 = -1.05e-6
    n3 = 1.6e-8
    n4 = -2.02e-6
    n5 = 15.868
    n6 = 0.01155
    n7 = -0.00423
    n8 = -4382.0
    n9 = 1.1455e6
    nsw = ne.evaluate(
        "n0 + (n1 + n2 * degC + n3 * degC**2) * psu + n4 * degC**2"
        "+ (n5 + n6 * psu + n7 * degC) / wlngth + n8 / wlngth**2"
        "+ n9 / wlngth**3"
    )

    # pure seawater
    nsw = ne.evaluate("nsw * n_air")
    dnds = ne.evaluate("(n1 + n2 * degC + n3 * degC**2 + n6 / wlngth) * n_air")

    return nsw, dnds


def flo_isotherm_compress(degC, psu):
    """
    Helper function for flo_zhang_scatter_coeffs
    @param degC in situ water temperature
    @param psu in site practical salinity
    @retval iso_comp seawater isothermal compressibility
    """
    # pure water secant bulk Millero (1980, Deep-sea Research)
    kw = ne.evaluate(
        "19652.21 + 148.4206 * degC - 2.327105 * degC**2"
        "+ 1.360477e-2 * degC**3 - 5.155288e-5 * degC**4"
    )

    # seawater secant bulk
    a0 = ne.evaluate(
        "54.6746 - 0.603459 * degC + 1.09987e-2 * degC**2" "- 6.167e-5 * degC**3"
    )
    b0 = ne.evaluate("7.944e-2 + 1.6483e-2 * degC - 5.3009e-4 * degC**2")
    ks = ne.evaluate("kw + a0 * psu + b0 * psu**1.5")

    # calculate seawater isothermal compressibility from the secant bulk
    iso_comp = ne.evaluate("1 / ks * 1e-5")  # unit is Pa

    return iso_comp


def flo_density_seawater(degC, psu):
    """
    Helper function for flo_zhang_scatter_coeffs
    @param degC in situ water temperature
    @param psu in site practical salinity
    @retval rho_sw density of seawater
    """
    # density of water and seawater,unit is Kg/m^3, from UNESCO,38,1981
    a0 = 8.24493e-1
    a1 = -4.0899e-3
    a2 = 7.6438e-5
    a3 = -8.2467e-7
    a4 = 5.3875e-9
    a5 = -5.72466e-3
    a6 = 1.0227e-4
    a7 = -1.6546e-6
    a8 = 4.8314e-4
    b0 = 999.842594
    b1 = 6.793952e-2
    b2 = -9.09529e-3
    b3 = 1.001685e-4
    b4 = -1.120083e-6
    b5 = 6.536332e-9

    # density for pure water
    rho_w = ne.evaluate(
        "b0 + b1 * degC + b2 * degC**2 + b3 * degC**3" "+ b4 * degC**4 + b5 * degC**5"
    )

    # density for pure seawater
    rho_sw = ne.evaluate(
        "rho_w + ((a0 + a1 * degC + a2 * degC**2"
        "+ a3 * degC**3 + a4 * degC**4) * psu"
        "+ (a5 + a6 * degC + a7 * degC**2) * psu**1.5 + a8 * psu**2)"
    )

    return rho_sw


def flo_scale_and_offset(counts_output, counts_dark, scale_factor):
    """
    Description:
        This scale and offset function is a simple numeric expression that can
        be applied to the CHLAFLO, CDOMFLO, FLUBSCT data products
    Implemented by:
        2014-01-30: Craig Risien. Initial Code
    Usage:
        value = flo_scale_and_offset(counts_output, counts_dark, scale_factor)
            where
        value = output value
        counts_output = measured sample output [counts]
        counts_dark = measured signal output of fluormeter in clean water with
                      black tape over the detector [counts]
        scale_factor = multiplier [units counts^-1]
    References:
        N/A
    """
    value = ne.evaluate("(counts_output - counts_dark) * scale_factor")
    return value


def flo_chla(counts_output, counts_dark, scale_factor):
    """
    Description:
        The OOI Level 1 Fluorometric Chlorophyll-a Concentration core data
        product is a measure of how much light has been re-emitted after
        being absorbed by Chlorophyll-a molecules found in all phytoplankton.
        By measuring the intensity and nature of this fluorescence,
        phytoplankton biomass can be estimated. The concentration of
        Chlorophyll-a is a proxy for the abundance of phytoplankton in the
        water column, and thus the amount of primary productivity that can be
        empirically achieved. Chlorophyll absorbs photons in the visible
        spectrum (400-700nm) and fluoresces visible blue light.
    Implemented by:
        2014-01-30: Craig Risien. Initial Code
    Usage:
        chla_conc = flo_chla(counts_output, counts_dark, scale_factor)
            where
        chla_conc = Fluorometric Chlorophyll-a Concentration (CHLAFLO_L1) [ug L^-1]
        counts_output = measured sample output (CHLAFLO_L0) [counts]
        counts_dark = measured signal output of fluormeter in clean water with
                      black tape over the detector [counts]
        scale_factor = multiplier [ug L^-1 counts^-1]
    References:
        OOI (2012). Data Product Specification for Fluorometric Chlorophyll-a
            Concentration. Document Control Number 1341-00530.
            https://alfresco.oceanobservatories.org/ (See: Company Home >> OOI
            >> Controlled >> 1000 System Level >>
            1341-00530_Data_Product_SPEC_CHLAFLO_OOI.pdf)
    """
    chla_conc = flo_scale_and_offset(counts_output, counts_dark, scale_factor)
    return chla_conc


def flo_cdom(counts_output, counts_dark, scale_factor):
    """
    Description:
        The OOI Level 1 Fluorometric CDOM concentration core data product is a
        measure of how much light has been re-emitted from refractory colored
        organic compounds found in the fluorometric pool of colored dissolved
        organic matter (CDOM) in seawater. This data product describes a
        measure of the amount of tannins (polyphenols that bind to proteins and
        other large molecules) or lignins (polymers of phenolic acids) from
        decaying plant material or byproducts from the decomposition of
        animals. It accounts for the tea-like color of some water masses. CDOM
        is not particulate, but water masses can contain both CDOM and
        turbidity. CDOM absorbs ultraviolet light and fluoresces visible blue
        light. The fluorescence of CDOM is used in many applications such as
        continuous monitoring of wastewater discharge, natural tracer of
        specific water bodies, ocean color research and the effect of CDOM on
        satellite imagery, and investigations of CDOM concentrations impacting
        light availability used for primary production.
    Implemented by:
        2014-01-30: Craig Risien. Initial Code
    Usage:
        cdom_conc = flo_cdom(counts_output, counts_dark, scale_factor)
            where
        cdom_conc = Fluorometric CDOM Concentration (CDOMFLO_L1) [ppb]
        counts_output = measured sample output (CDOMFLO_L0) [counts]
        counts_dark = measured signal output of fluormeter in clean water with
                      black tape over the detector [counts]
        scale_factor = multiplier [ppb counts^-1]
    References:
        OOI (2012). Data Product Specification for Fluorometric CDOM
            Concentration. Document Control Number 1341-00550.
            https://alfresco.oceanobservatories.org/ (See: Company Home >> OOI
            >> Controlled >> 1000 System Level >>
            1341-00550_Data_Product_SPEC_CDOMFLO_OOI.pdf)
    """
    cdom_conc = flo_scale_and_offset(counts_output, counts_dark, scale_factor)
    return cdom_conc


def flo_beta(counts_output, counts_dark, scale_factor):
    """
    Description:
        This function calculates FLUBSCT_L1, the value for the volume scattering
        function measured at the effective measurement angle determined by the
        sensor geometry of the particular instrument used. Most of the FLORD,
        FLORT, and FLNTU instruments measure backscattering at a wavelength of
        700nm (the exception is the FLORT series O ECO-BB3 instrument). For more
        information see the Notes section of the function flo_bback_total in this
        module.
    Implemented by:
        2014-01-30: Craig Risien. Initial Code
        2015-10-23: Russell Desiderio. Revised documentation.
    Usage:
        beta = flo_flubsct(counts_output, counts_dark, scale_factor)
            where
        beta = value of the volume scattering function measured at the effective
               (centroid) backscatter angle of the measuring instrument at its
               measurement wavelength (usually 700nm) (FLUBSCT_L1) [m^-1 sr^-1]
        counts_output = measured sample output (FLUBSCT_L0) [counts]
        counts_dark = measured signal output of fluorometer in clean water
                      with black tape over the detector [counts]
        scale_factor = multiplier [m^-1 sr^-1 counts^-1]
    References:
        OOI (2012). Data Product Specification for Fluorometric Chlorophyll-a
            Concentration. Document Control Number 1341-00540 (version 1-05).
            https://alfresco.oceanobservatories.org/ (See: Company Home >> OOI
            >> Controlled >> 1000 System Level >>
            1341-00540_Data_Product_SPEC_FLUBSCT_OOI.pdf)
    """
    beta = flo_scale_and_offset(counts_output, counts_dark, scale_factor)
    return beta
