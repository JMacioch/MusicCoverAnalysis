import React, { useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useDispatch } from "react-redux";
import { Navbar, Nav, Container, Row, Col, Button } from "react-bootstrap";

export default function Main({ onStart }) {
  const navigate = useNavigate();

  useEffect(() => {
    const handleKeyDown = (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        navigate("/survey");
        if (onStart) {
          console.log("Button clicked");
          onStart();
        }
      }
    };
    window.addEventListener("keydown", handleKeyDown);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [navigate, onStart]);

  function startSurvey() {
    onStart();
    navigate("/survey");
  }

  return (
    <div>
      {/* Navbar Section */}
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
          <p className="lead">
            You will be asked 15 questions one after another. You can review and
            change answers before you finish the survey. Provided answers will
            be anonymous.
          </p>
          <Button variant="primary" size="lg" onClick={startSurvey}>
            Start Survey
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
