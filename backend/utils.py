from sqlalchemy import and_, exists

import models


def is_exercise_assigned_to_user(db, user_id, exercise_id):
    return db.query(
        exists().where(
            and_(
                models.AssignedExercise.user_id == user_id,
                models.AssignedExercise.exercise_id == exercise_id,
            )
        )
    ).scalar()
