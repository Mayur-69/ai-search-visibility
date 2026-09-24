from aiv.database import get_session
from aiv.models import Page
from sqlalchemy import delete

with get_session() as session:
    session.execute(delete(Page))
    session.commit()
    print("Cleared pages")