
#include "one_action_entropy_reduction_planner.hpp"
#include <ruler-point-process/ruler_point_process.hpp>
#include <igmm-point-process/igmm_point_process.hpp>
#include <math-core/geom.hpp>
#include <math-core/io.hpp>
#include <math-core/utils.hpp>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <boost/pointer_cast.hpp>
#include <stdexcept>


#define VERBOSE false
#define PRINT_WARNINGS true

using namespace point_process_core;
using namespace math_core;

namespace planner_core {

  //=======================================================================
  //=======================================================================
  //=======================================================================

  double
  expected_posterior_entropy_difference
  ( const boost::shared_ptr<mcmc_point_process_t>& point_process,
    const std::vector<nd_point_t>& new_obs,
    const entropy_estimator_parameters_t& ent_params )
  {
    boost::shared_ptr<ruler_point_process::ruler_point_process_t> rp
      = boost::dynamic_pointer_cast<ruler_point_process::ruler_point_process_t >( point_process );
    if( rp ) {

      // this has a close form solution!
      return rp->expected_posterior_entropy_difference( new_obs );
      
    } 


    boost::shared_ptr<igmm_point_process::igmm_point_process_t> igmm
      = boost::dynamic_pointer_cast<igmm_point_process::igmm_point_process_t >( point_process );
    if( igmm ) {

      // this has a close form solution!
      return igmm->expected_posterior_entropy_difference( new_obs );
      
    } 



    // we have to approximate this with samples
    // for now throw an exception
    throw( std::runtime_error( "entropy difference not implemented for anything but the ruler_point_process_t!" ) );
    return 0;
  }

  //=======================================================================

  void
  bin_points( const std::vector<nd_point_t>& points,
	      marked_grid_t<std::vector<nd_point_t> >& bins )
  {
    for( nd_point_t p : points ) {
      marked_grid_cell_t cell = bins.cell( p );
      if( !bins( cell ) ) {
	bins.set( cell, std::vector<nd_point_t>() );
      }
      std::vector<nd_point_t> x = *bins( cell );
      x.push_back( p );
      bins.set( cell, x );
    }
  }

  //=======================================================================

  void
  one_action_entropy_reduction_planner_t
  ::calculate_expected_entropy_reduction( marked_grid_t<double>& red_grid ) const
  {

    // calculate the grid cells which have not yet been visited and so 
    // are potential actions
    std::vector<marked_grid_cell_t> action_cells;
    std::vector<marked_grid_cell_t> all_cells = _visited_grid.all_cells();
    for( marked_grid_cell_t cell : all_cells ) {
      if( !_visited_grid(cell) ||
	  *_visited_grid(cell) == false ) {
	action_cells.push_back( cell );
      }
    }

    
    // sample a number of worlds from the model and use those to
    // calculate the expected posterior entroipy for every grid cell
    for( size_t world_i = 0; world_i < _sampler_planner_params.num_samples_of_point_sets; ++world_i ) {

      // sample a world
      std::vector<nd_point_t> world = _point_process->sample_and_step();

      // bin the points into the cells
      marked_grid_t<std::vector<nd_point_t> > binned_world 
	= red_grid.copy_structure<std::vector<nd_point_t> >();
      bin_points( world, binned_world );

      // calculate the entropy difference for every binned cells
      for( marked_grid_cell_t cell : action_cells ) {
	double e_diff = 0;
	if( binned_world(cell) ) {
	  e_diff = 
	    expected_posterior_entropy_difference( _point_process,
						   *binned_world( cell ),
						   _entropy_params );

	  if( VERBOSE ) {
	    std::cout << "  sample[" << world_i << "].cell{" << cell << "} ediff= " << e_diff << " #" << binned_world(cell)->size() << std::endl;
	  }

	}
	if( !red_grid(cell) ) {
	  red_grid.set( cell, 0.0 );
	}
	red_grid.set( cell, e_diff + (*red_grid(cell)) );
      }
    }
  }

  //=======================================================================

  marked_grid_cell_t 
  one_action_entropy_reduction_planner_t::
  choose_next_observation_cell()
  {

    // calculate the expected entropy reduction for all possible cells
    marked_grid_t<double> expected_entropy_reduction
      = _visited_grid.copy_structure<double>();
    calculate_expected_entropy_reduction( expected_entropy_reduction );
    
    // now find the maximum reduction cell
    marked_grid_cell_t max_cell;
    double max_red = - std::numeric_limits<double>::infinity();
    for( marked_grid_cell_t cell : expected_entropy_reduction.all_cells() ) {
      if( expected_entropy_reduction( cell ) ) {
	double r = *expected_entropy_reduction(cell);
	if( r > max_red ) {
	  max_cell = cell;
	  max_red = r;
	}
      }
    }

    if( VERBOSE ) {
      std::cout << "Max Ent Reduction: " << max_red << " cell: " << max_cell << std::endl;
    }

    // make sure we have found a reducing cell and return it
    if( max_red > -std::numeric_limits<double>::infinity() 
	&& ( _visited_grid(max_cell) == false 
	     || *_visited_grid(max_cell) == false ) ) {
      return max_cell;
    }

    // Ok, what if the max cell  is invalid for some reason
    // Then jsut choose the first non-marked cell
    std::vector<marked_grid_cell_t> all_cells = _visited_grid.all_cells();
    for( size_t i = 0; i < all_cells.size(); ++i ) {
      if( !_visited_grid( all_cells[i] ) ) {
	return all_cells[i];
      }
    }
    
    // getting here is unfortunate!
    if( PRINT_WARNINGS ) {
      std::cout << "AHHHHH! Why is there no unmarkes grid cell!" << std::endl;
    }
    return _visited_grid.cell( _current_position );

  }

  //=======================================================================


  one_action_entropy_reduction_planner_t::
  one_action_entropy_reduction_planner_t
  ( const boost::shared_ptr<mcmc_point_process_t>& process,
    const grid_planner_parameters_t& planner_params,
    const entropy_estimator_parameters_t& entropy_params,
    const sampler_planner_parameters_t& sampler_planner_params)
    : _point_process( process ),
      _planner_params( planner_params ),
      _entropy_params( entropy_params ),
      _sampler_planner_params( sampler_planner_params )
  {
    
    // create the grids with no marks using the process window
    _visited_grid = marked_grid_t<bool>( _point_process->window(),
					 _planner_params.grid_cell_size );
    _negative_observation_grid = marked_grid_t<bool>( _point_process->window(),
						      _planner_params.grid_cell_size );
    
  }

  //=======================================================================

  //=======================================================================
  
  //=======================================================================


  void 
  one_action_entropy_reduction_planner_t::print_shallow_trace( std::ostream& out ) const
  {

    // out << "{ \"object_class\" : \"one_action_entropy_reduction_planner_t\" , ";
    // out << "  \"current_position\" : " << to_json( _current_position ) << " , ";
    // out << "  \"planner_params\" : " << to_json( _planner_params ) << " , ";
    // out << "  \"entropy_params\" : " << to_json( _entropy_params ) << " , ";
    // out << "  \"sample_planner_params\" : " << to_json( _sampler_planner_params ) << " , ";
    // out << "  \"observations\" : [ ";
    // for( size_t i = 0 ; i < _observations.size(); ++i ) {
    //   out << to_json( _observations[i]  );
    //   if( i < _observations.size() - 1 ) {
    // 	out << ",";
    //   }
    // }
    // out << "], ";
    // out << "  \"visited_grid\" : " << to_json( _visited_grid ) << " , ";
    // out << "  \"negative_observation_grid\" : " << to_json( _negative_observation_grid ) << "";
    // out << "} ";
    
  }

  //=======================================================================
  //=======================================================================
  //=======================================================================
  //=======================================================================

  one_action_entropy_reduction_planner_t::~one_action_entropy_reduction_planner_t()
  {
  }

  //=======================================================================
  //=======================================================================
  //=======================================================================
  //=======================================================================
  //=======================================================================
  //=======================================================================
  //=======================================================================
  //=======================================================================
  //=======================================================================
  


}
