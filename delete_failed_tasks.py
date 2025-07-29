#!/usr/bin/env python3
import sqlite3

def delete_failed_tasks():
    db_path = 'trellis_simulation_outputs/trellis_simulator_tasks.db'
    
    # These are the tasks that failed with HTTP 500 errors
    failed_task_ids = ['sim_68', 'sim_69', 'sim_70']
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Delete the specific failed tasks
        for task_id in failed_task_ids:
            cursor.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))
            deleted_count = cursor.rowcount
            print(f"Deleted {deleted_count} record(s) for task {task_id}")
        
        conn.commit()
        
        # Verify deletion
        cursor.execute("SELECT COUNT(*) FROM tasks")
        remaining_count = cursor.fetchone()[0]
        print(f"\nRemaining tasks in database: {remaining_count}")
        
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    delete_failed_tasks() 
