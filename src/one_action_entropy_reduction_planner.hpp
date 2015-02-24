

#if !defined( __PLANNER_CORE_one_action_entropy_reduction_planner_HPP__ )
#define __PLANNER_CORE_one_action_entropy_reduction_planner_HPP__


#include "planner.hpp"
#include <point-process-core/entropy.hpp>

#define SAMPLING_VERBOSE false

namespace planner_core {


  using namespace point_process_core;
  using namespace math_core;


  
  // Description:
  // A grid planner which tries to maximize the entropy change
  // for the next action
  class one_action_entropy_reduction_planner_t
    : public grid_planner_t
  {
  public:
    
    // Description:
    // Creates a new planner with the given point process as model
    // and entropy parameters
    one_action_entropy_reduction_planner_t
    ( const boost::shared_ptr<mcmc_point_process_t>& process,
      const grid_planner_parameters_t& planner_params,
      const entropy_estimator_parameters_t& entropy_params,
      const sampler_planner_parameters_t& sampler_planner_params );


    virtual ~one_action_entropy_reduction_planner_t();


    // Description:
    // Returns the observation grid/ visited grid which has marked as true
    // all cells which have been observed
    virtual
    marked_grid_t<bool>
    visited_grid() const
    {
      return _visited_grid;
    }

    // Description:
    // Adds a given cell as already visisted.
    virtual
    void add_visited_cell( const marked_grid_cell_t& cell )
    {
      _visited_grid.set( cell, true );
    }
    
    // Description:
    // Add a negative region
    virtual
    void add_negative_observation( const marked_grid_cell_t& cell )
    {
      _negative_observation_grid.set( cell, true );
      _point_process->add_negative_observation( _negative_observation_grid.region(cell) );
      _point_process->mcmc( _planner_params.update_model_mcmc_iterations, SAMPLING_VERBOSE );
    }

    // Description:
    // Adds an "Empty" region, which is not a negative observation, rather
    // a region within an observed cell with points that has no points
    // hence the empty space of a cell with points
    virtual
    void add_empty_region( const nd_aabox_t& region )
    {
      _point_process->add_negative_observation( region );
    }


    // Description:
    // Add a set of observation poitns
    virtual
    void add_observations( const std::vector<math_core::nd_point_t>& obs )
    {
      _point_process->add_observations( obs );
      _point_process->mcmc( _planner_params.update_model_mcmc_iterations, SAMPLING_VERBOSE );
      _observations.insert( _observations.end(), obs.begin(), obs.end() );
    }

    // Description:
    // Returns the next observation to take
    virtual
    marked_grid_cell_t choose_next_observation_cell();

    // Description:
    // Sets the current position
    virtual
    void set_current_position( const nd_point_t& pos )
    { 
      _current_position = pos;
    }

    // Description:
    // Retruns the observations
    virtual
    std::vector<math_core::nd_point_t> observations() const
    {
      return _observations;
    }

    // Description:
    // Returns true iff all cells have been visited
    virtual
    bool all_cells_visited() const
    {
      // TODO:
      // Write this!
      return false;
    }

    // Description:
    // Print outs a shallow trace for this planner
    virtual
    void print_shallow_trace( std::ostream& out ) const;

    // Description:
    // print shallow trace of the model
    virtual
    void print_model_shallow_trace( std::ostream& out ) const
    {
      if( _point_process ) {
	_point_process->print_shallow_trace( out );
      }
    }

    // Description:
    // Get/Set the grid_planner_parameters_t parameters
    virtual
    grid_planner_parameters_t get_grid_planner_parameters() const
    {
      return _planner_params;
    }
    virtual
    void set_grid_planner_parameters( const grid_planner_parameters_t& p )
    {
      _planner_params = p;
    }

    // Description:
    // Plot this planner.
    virtual
    std::string
    plot( const std::string& title ) const
    { return "UNK"; }


    // Descriotion:
    // Plot all of the stored next_cell dataseries into a time plot
    virtual
    std::string
    plot_all_next_cell_dist( const std::string& title ) const
    { return "UNK<SERIES>"; }

  public:
    
    // Description:
    // Calculates the expected posterior entropy using samples of worlds
    // from our own point process model.
    void
    calculate_expected_entropy_reduction
    ( marked_grid_t<double>& red_grid ) const;

    // Description:
    // the current position
    nd_point_t _current_position;
    
    // Description:
    // The point process used as model
    boost::shared_ptr<mcmc_point_process_t> _point_process;

    // Description:
    // The parameters for planning and entropy computations
    grid_planner_parameters_t _planner_params;
    entropy_estimator_parameters_t _entropy_params;
    sampler_planner_parameters_t _sampler_planner_params;

    // Description:
    // The observations so far
    std::vector<math_core::nd_point_t> _observations;
    
    // Description:
    // The grid of visited cells
    marked_grid_t<bool> _visited_grid;
        
    // Description:
    // The grid of negative observation
    marked_grid_t<bool> _negative_observation_grid;


  };

}

#endif
