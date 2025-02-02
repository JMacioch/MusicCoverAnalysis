import React, { useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useDispatch, useSelector } from "react-redux";

/** import actions */
import { resetAllAction } from "../redux/question_reducer";
import { resetResultAction } from "../redux/result_reducer";
import { usePublishResult } from "../hooks/SetResult";
import { Navbar, Nav, Container, Row, Col, Button } from "react-bootstrap";
export default function Result() {
  const dispatch = useDispatch();
  const navigate = useNavigate();

  const { queue } = useSelector((state) => state.questions);
  const { result } = useSelector((state) => state.result);

  useEffect(() => {
    const handleKeyDown = (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        navigate("/");
        onRestart();
      }
    };
    window.addEventListener("keydown", handleKeyDown);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [navigate, onRestart]);

  /** store user result */
  usePublishResult({
    result,
  });

  function onRestart() {
    dispatch(resetResultAction());
    dispatch(resetAllAction());
    navigate("/");
  }

  return (
    <div>
      {/* Always Collapsed Navbar */}
      <Navbar bg="dark" variant="dark" expand={false}>
        <Container>
          <Navbar.Brand>
            <Link class="navbar-brand" to="/">
              Survey Application
            </Link>
          </Navbar.Brand>
        </Container>
      </Navbar>

      {/* Header Section */}
      <header className="bg-white text-center py-5">
        <Container>
          <h1 className="display-4">Survey Application</h1>
          <p className="lead">Thank you for participating in the survey.</p>
          <Button variant="primary" size="lg" onClick={onRestart}>
            Restart
          </Button>
        </Container>
      </header>

      {/* Footer Section */}
      <footer className="mt-5 mb-5 text-center text-muted">
        &copy; 2024 Survey Application. All rights reserved.
      </footer>
    </div>
  );
}
