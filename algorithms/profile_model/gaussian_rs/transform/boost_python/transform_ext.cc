/*
 * transform_ext.cc
 *
 *  Copyright (C) 2013 Diamond Light Source
 *
 *  Author: James Parkhurst
 *
 *  This code is distributed under the BSD license, a copy of which is
 *  included in the root directory of this package.
 */
#include <boost/python.hpp>
#include <boost/python/def.hpp>
#include <dials/algorithms/profile_model/gaussian_rs/transform/beam_vector_map.h>
#include <dials/algorithms/profile_model/gaussian_rs/transform/index_generator.h>
#include <dials/algorithms/profile_model/gaussian_rs/transform/map_frames.h>
#include <dials/algorithms/profile_model/gaussian_rs/transform/transform.h>

namespace dials {
namespace algorithms {
namespace profile_model {
namespace gaussian_rs {
namespace transform {
namespace boost_python {

  using namespace boost::python;

  CoordinateGenerator* make_coordinate_generator(
      const CoordinateSystem &cs,
      int x0, int y0,
      af::versa< vec3<double>, scitbx::af::flex_grid<> > s1_map) {
    return new CoordinateGenerator(cs, x0, y0,
      af::versa< vec3<double>, af::c_grid<2> >(s1_map.handle(),
        af::c_grid<2>(s1_map.accessor())));
  }

  GridIndexGenerator* make_grid_index_generator(const CoordinateSystem &cs,
      int x0, int y0, vec2<double> step_size, std::size_t grid_half_size,
      af::versa< vec3<double>, scitbx::af::flex_grid<> > s1_map) {
    return new GridIndexGenerator(cs, x0, y0, step_size, grid_half_size,
      af::versa< vec3<double>, af::c_grid<2> >(
        s1_map.handle(), af::c_grid<2>(s1_map.accessor())));
  }

  template <typename FloatType>
  boost::shared_ptr< TransformSpec<FloatType> >
    init_forward_from_experiment(
      object experiment, double sigma_b, double sigma_m,
      double nsigma, std::size_t grid_size) {
    return boost::shared_ptr< TransformSpec<FloatType> >(
      new TransformSpec<FloatType>(
        extract<Beam>(experiment.attr("beam")),
        extract<Detector>(experiment.attr("detector")),
        extract<Goniometer>(experiment.attr("goniometer")),
        extract<Scan>(experiment.attr("scan")),
        sigma_b, sigma_m, nsigma, grid_size));
  }

  template <typename FloatType>
  void forward_wrapper(const char *name) {

    typedef Forward<FloatType> ForwardType;
    typedef TransformSpec<FloatType> TransformSpecType;

    class_<TransformSpecType>("TransformSpec", no_init)
      .def(init<const Beam &, const Detector &,
                const Goniometer &, const Scan &,
                double, double, double, std::size_t>())
      .def("__init__", make_constructor(
        &init_forward_from_experiment<FloatType>))
      .def("m2", &TransformSpecType::m2)
      .def("s0", &TransformSpecType::s0)
      .def("image_size", &TransformSpecType::image_size)
      .def("grid_size", &TransformSpecType::grid_size)
      .def("step_size", &TransformSpecType::step_size)
      .def("grid_centre", &TransformSpecType::grid_centre)
      .def("s1_map", &TransformSpecType::s1_map);

    class_<ForwardType>(name, no_init)
      .def(init<const TransformSpec<FloatType>&,
                const vec3<double>&, double, int6, std::size_t,
                const af::const_ref< FloatType, af::c_grid<3> >&,
                const af::const_ref< bool, af::c_grid<3> >& >())
      .def(init<const TransformSpec<FloatType>&,
                const vec3<double>&, double, int6, std::size_t,
                const af::const_ref< FloatType, af::c_grid<3> >&,
                const af::const_ref< FloatType, af::c_grid<3> >&,
                const af::const_ref< bool, af::c_grid<3> >& >())
      .def(init<const TransformSpec<FloatType>&,
                const CoordinateSystem&, int6, std::size_t,
                const af::const_ref< FloatType, af::c_grid<3> >&,
                const af::const_ref< bool, af::c_grid<3> >& >())
      .def(init<const TransformSpec<FloatType>&,
                const CoordinateSystem&, int6, std::size_t,
                const af::const_ref< FloatType, af::c_grid<3> >&,
                const af::const_ref< FloatType, af::c_grid<3> >&,
                const af::const_ref< bool, af::c_grid<3> >& >())
      .def("profile", &ForwardType::profile)
      .def("background", &ForwardType::background)
      .def("zfraction", &ForwardType::zfraction);
  }

  BOOST_PYTHON_MODULE(dials_algorithms_profile_model_gaussian_rs_transform_ext)
  {
    af::versa< vec3<double>, af::c_grid<2> > (*overload1)(const Panel&,
      const Beam&, std::size_t, bool) = &beam_vector_map;
    af::versa< vec3<double>, af::c_grid<2> > (*overload2)(const Panel&,
      const Beam&, bool) = &beam_vector_map;
    af::versa< vec3<double>, af::c_grid<2> > (*overload3)(const Panel&,
      const Beam&) = &beam_vector_map;

    def("beam_vector_map", overload1, (
      arg("detector"),
      arg("beam"),
      arg("n_div"),
      arg("corners")));
    def("beam_vector_map", overload2, (
      arg("detector"),
      arg("beam"),
      arg("corners")));
    def("beam_vector_map", overload3, (
      arg("detector"),
      arg("beam")));

    class_<CoordinateGenerator>("CoordinateGenerator", no_init)
      .def("__init__", make_constructor(
        &make_coordinate_generator,
        default_call_policies(), (
          arg("cs"),
          arg("x0"),
          arg("y0"),
          arg("s1_map"))))
      .def("__call__", &CoordinateGenerator::operator());

    class_<GridIndexGenerator>("GridIndexGenerator", no_init)
      .def("__init__", make_constructor(
        &make_grid_index_generator,
        default_call_policies(), (
          arg("cs"),
          arg("x0"),
          arg("y0"),
          arg("step_size"),
          arg("grid_half_size"),
          arg("s1_map"))))
      .def("__call__", &GridIndexGenerator::operator());

    typedef MapFramesForward<> MapFramesForwardType;
    typedef MapFramesReverse<> MapFramesReverseType;

    class_<MapFramesForwardType>(
        "MapFramesForward", no_init)
      .def(init<double, double, double, double, int>((
          arg("starting_angle"),
          arg("oscillation"),
          arg("mosaicity"),
          arg("n_sigma"),
          arg("grid_size_e3"))))
      .def("__call__",
        &MapFramesForwardType::operator(), (
          arg("frames"),
          arg("phi"),
          arg("zeta")));

    class_<MapFramesReverseType>(
        "MapFramesReverse", no_init)
      .def(init<double, double, double, double, int>((
          arg("starting_angle"),
          arg("oscillation"),
          arg("mosaicity"),
          arg("n_sigma"),
          arg("grid_size_e3"))))
      .def("__call__",
        &MapFramesReverseType::operator(), (
          arg("frames"),
          arg("phi"),
          arg("zeta")));

    forward_wrapper<double>("Forward");
  }

}}}}}} // namespace dials::algorithms::reflexion_basis::transform::boost_python
